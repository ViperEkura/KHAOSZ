import math

def get_sgdr_schedule(warning_step, cycle_length, min_rate=0.1, T_mult=2):
    def sgdr_schedule(now_iter):
        if now_iter < warning_step:
            return max(min_rate, now_iter / warning_step)
            
        adjusted_iter = now_iter - warning_step
        total_cycles, current_cycle = 0, 0
        while adjusted_iter >= cycle_length * (T_mult ** total_cycles):
            current_cycle += 1
            total_cycles += 1
        
        cycle_start = sum(cycle_length * (T_mult ** i) for i in range(current_cycle))
        cycle_pos = adjusted_iter - cycle_start
        
        cycle_length_current = cycle_length * (T_mult ** current_cycle)
        return (min_rate, 0.5 * (1 + math.cos(math.pi * cycle_pos / cycle_length_current)))
    
    return sgdr_schedule


def get_cosine_warmup_schedule(warning_step, lr_decay_iters, min_rate=0.1):
    def cosine_warmup_schedule(now_iter):
        if now_iter <= warning_step:
            return max(min_rate, now_iter / warning_step)
        else:
            rate = (now_iter - warning_step) / (lr_decay_iters - warning_step)
            return max(min_rate, 0.5 * (1.0 + math.cos(math.pi * rate)))
    
    return cosine_warmup_schedule