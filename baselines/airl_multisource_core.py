"""多源 AIRL：判别与策略更新与 baselines/adv_smm.py 中 AIRL 分支一致（disc_forward_airl / get_reward）。"""
import numpy as np
import torch
from torch import autograd


def process_airl_expert_buffer(expert_trajs, expert_action_trajs):
    """(N,T,d) 状态与 (N,T,a) 动作 -> (obs, act, obs2) 展平，与 AdvSMM.process_target_state_buffer 一致。"""
    obs_l, act_l, obs2_l = [], [], []
    for idx in range(len(expert_trajs)):
        obs_l.append(expert_trajs[idx][:-1])
        act_l.append(expert_action_trajs[idx][1:])
        obs2_l.append(expert_trajs[idx][1:])
    obs = np.concatenate(obs_l, axis=0)
    actions = np.concatenate(act_l, axis=0)
    obs2 = np.concatenate(obs2_l, axis=0)
    return obs, actions, obs2


def disc_forward_airl(
    p_obs,
    p_act,
    p_obs_2,
    e_obs,
    e_act,
    e_obs_2,
    reward_model,
    value_model,
    agent,
    gamma,
):
    """与 AdvSMM.disc_forward_airl 一致 + 数值保护，避免 NaN/Inf 导致 BCELoss 报错。"""
    obs = torch.cat([e_obs, p_obs], dim=0)
    act = torch.cat([e_act, p_act], dim=0)
    obs_2 = torch.cat([e_obs_2, p_obs_2], dim=0)

    reward = reward_model(obs)
    cur_val = value_model(obs)
    next_val = value_model(obs_2)

    log_p = reward + gamma * next_val - cur_val
    with torch.no_grad():
        log_q = agent.ac.log_prob(obs, act)
        log_q = log_q.unsqueeze(1)
        baseline = torch.max(log_p, log_q)

    log_p = log_p - baseline
    log_q = log_q - baseline

    # === 新增：数值稳定保护（关键）===
    log_p = torch.clamp(log_p, -20.0, 20.0)
    log_q = torch.clamp(log_q, -20.0, 20.0)
    disc_logits = torch.exp(log_p) / (torch.exp(log_p) + torch.exp(log_q) + 1e-8)
    disc_logits = torch.clamp(disc_logits, 0.0, 1.0)   # 强制落在 BCELoss 合法区间

    disc_preds = (disc_logits > 0.5).type(disc_logits.data.type())
    return disc_logits, disc_preds


def get_airl_reward_batch(
    obs,
    obs2,
    reward_model,
    value_model,
    gamma,
    reward_scale,
    airl_shaping,
):
    """与 AdvSMM.get_reward 在 mode=='airl' 时一致（无 nan_to_num、无 rew_clip）。"""
    reward_model.eval()
    value_model.eval()
    with torch.no_grad():
        rewards = reward_model(obs)
        if airl_shaping:
            rewards = rewards + gamma * value_model(obs2) - value_model(obs)
        rewards = rewards.view(-1)
        rewards = torch.clamp(rewards, -50.0, 50.0)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=50.0, neginf=-50.0)
    reward_model.train()
    value_model.train()
    rewards = rewards * reward_scale
    return rewards


def airl_disc_training_step(
    replay_buffer,
    expert_obs,
    expert_act,
    expert_obs2,
    reward_model,
    value_model,
    agent,
    disc_optimizer,
    bce,
    bce_targets,
    disc_optim_batch_size,
    gamma,
    device,
    use_grad_pen,
    grad_pen_weight,
):
    sampled = replay_buffer.sample_batch(disc_optim_batch_size)
    policy_state = sampled["obs"]
    policy_action = sampled["act"]
    policy_next_state = sampled["obs2"]

    ei = np.random.choice(len(expert_obs), size=disc_optim_batch_size)
    expert_state = torch.FloatTensor(expert_obs[ei]).to(device)
    expert_action = torch.FloatTensor(expert_act[ei]).to(device)
    expert_next_state = torch.FloatTensor(expert_obs2[ei]).to(device)

    disc_optimizer.zero_grad()
    disc_logits, _ = disc_forward_airl(
        policy_state,
        policy_action,
        policy_next_state,
        expert_state,
        expert_action,
        expert_next_state,
        reward_model,
        value_model,
        agent,
        gamma,
    )
    disc_ce_loss = bce(disc_logits, bce_targets)

    disc_grad_pen_loss = 0.0
    if use_grad_pen:
        eps = torch.rand((disc_optim_batch_size, 1)).to(device)
        interp_obs = eps * expert_state + (1 - eps) * policy_state
        interp_obs = interp_obs.detach()
        interp_obs.requires_grad_(True)
        gradients = autograd.grad(
            outputs=reward_model(interp_obs).sum(),
            inputs=[interp_obs],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        eps2 = torch.rand((disc_optim_batch_size, 1)).to(device)
        interp_obs2 = eps2 * expert_state + (1 - eps2) * policy_state
        interp_obs2 = interp_obs2.detach()
        interp_obs2.requires_grad_(True)
        gradients2 = autograd.grad(
            outputs=value_model(interp_obs2).sum(),
            inputs=[interp_obs2],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = gradient_penalty + ((gradients2.norm(2, dim=1) - 1) ** 2).mean()
        disc_grad_pen_loss = gradient_penalty * grad_pen_weight

    disc_total_loss = disc_ce_loss + disc_grad_pen_loss
    torch.nn.utils.clip_grad_norm_(list(reward_model.parameters()) + list(value_model.parameters()), max_norm=5.0)
    disc_total_loss.backward()
    disc_optimizer.step()
    return np.array(
        [
            disc_total_loss.item(),
            disc_ce_loss.item(),
            disc_total_loss.item() - disc_ce_loss.item(),
        ]
    )


def airl_policy_training_step(
    replay_buffer,
    agent,
    policy_optim_batch_size,
    reward_model,
    value_model,
    gamma,
    reward_scale,
    airl_shaping,
):
    policy_batch = replay_buffer.sample_batch(policy_optim_batch_size)
    obs, obs2 = policy_batch["obs"], policy_batch["obs2"]
    policy_batch["rew"] = get_airl_reward_batch(
        obs,
        obs2,
        reward_model,
        value_model,
        gamma,
        reward_scale,
        airl_shaping,
    )
    return agent.update(policy_batch)
