import wandb

run = wandb.init(entity='linusbao', project="test", group="round4", name="init_test")
run.log({"wandb_test":123})
run.finish()