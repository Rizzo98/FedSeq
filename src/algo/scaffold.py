from src.algo import FedAvg
import logging
import torch
import os

log = logging.getLogger(__name__)


class SCAFFOLD(FedAvg):
    def __init__(self, model_info, params, device: str, dataset: str,
                 output_suffix: str, savedir: str, writer=None, wandbConf=None):
        assert params.optim.type == "scaf", "SCAFFOLD must use its optimizer"
        assert params.client.classname == "SCAFFOLDClient", "SCAFFOLD must use SCAFFOLDClient"
        assert params.center_server.classname == "SCAFFOLDCenterServer", "SCAFFOLD must use SCAFFOLDCenterServer"
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf=wandbConf)

    def send_data(self, clients):
        for client in clients:
            client.server_controls = self.center_server.send_controls()
            client.model = self.center_server.send_model()

    def load_from_checkpoint(self):
        for client in self.clients:
            client_path = client.client_path
            if os.path.exists(client_path):
                client.controls = torch.load(client.client_path)
        super().load_from_checkpoint()
