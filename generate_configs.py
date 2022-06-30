import os
import yaml

i = 1
for expander_factor in [1, 2, 4]:
    for pretrain_epoch, warmup_epoch in [
        (10, 2),
        (20, 4),
        (100, 10),
    ]:
        for pretrain_batch_size in [100, 200, 400]:
            for encoder_start_lr in [5e-4, 1e-3, 5e-3]:
                for dropout_rate_1, dropout_rate_2 in [
                    (0.25, 0.25),
                    (0.5, 0.5),
                    (0.75, 0.75),
                    (0.5, 1.0),
                ]:
                    for sim_coeff, std_coeff, cov_coeff in [
                        (25.0, 25.0, 1.0),
                        (12.5, 12.5, 1.0),
                    ]:
                        comment = f"CM_dropout_{i}"
                        comment = comment.replace(" ", "").replace("[", "").replace("]", "").replace(",", "_").replace("'", "")
                        config = {
                            "comment": comment,
                            "expander_factor": expander_factor,
                            "pretrain_epoch": pretrain_epoch,
                            "warmup_epoch": warmup_epoch,
                            "pretrain_batch_size": pretrain_batch_size,
                            "encoder_start_lr": encoder_start_lr,
                            "dropout_rate_1": dropout_rate_1,
                            "dropout_rate_2": dropout_rate_2,
                            "sim_coeff": sim_coeff,
                            "std_coeff": std_coeff,
                            "cov_coeff": cov_coeff,
                        }
                        with open(os.path.join("configs", f"experiment_{i}.yml"), 'w') as conf_file:
                            yaml.dump(config, conf_file)
                    
                        i += 1

                