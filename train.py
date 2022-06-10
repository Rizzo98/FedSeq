import logging, hydra, os
from omegaconf import OmegaConf, DictConfig
from src.utils import test_on_data
from src.utils import seed_everything, CustomSummaryWriter, WanDBSummaryWriter
from src.algo import *

log = logging.getLogger(__name__)

def create_model(cfg: DictConfig, writer) -> Algo:
    method: Algo = eval(cfg.algo.classname)(model_info=cfg.model, device=cfg.device, writer=writer,
                                            dataset=cfg.dataset, params=cfg.algo.params,
                                            savedir=cfg.savedir, output_suffix=cfg.output_suffix,
                                            wandbConf = cfg.wandb)
    if cfg.wandb.restart_from_run is not None:
        method.load_from_checkpoint()
    return method

@hydra.main(config_path="config", config_name="config",version_base='1.1')
def main(cfg: DictConfig):
    os.chdir(cfg.root)
    seed_everything(cfg.seed)
    log.info("\n" + OmegaConf.to_yaml(cfg))
    
    writer = WanDBSummaryWriter(cfg)
    writer.set_config(dict(cfg))
    
    model: Algo = create_model(cfg, writer)
    if cfg.do_train:
        model.fit(cfg.n_round)
    
    if cfg.test_on_data.dataset!=None:
        test_on_data(cfg.test_on_data,model.center_server.model,
            cfg.do_train, cfg.device, writer)

if __name__ == "__main__":
    main()
