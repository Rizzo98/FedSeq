from src.algo import FedSeq, SCAFFOLD


class FedSeqSCAFFOLD(FedSeq, SCAFFOLD):
    def send_data(self, clients):
        # clients are actually superclients
        for sp in clients:
            for client in sp.clients:
                client.server_controls = self.center_server.send_controls()
            sp.model = self.center_server.send_model()

    def aggregate(self, clients):
        # extract last client from each superclient
        scaffold_clients = []
        for sp in self.selected_clients:
            scaffold_clients.append(sp.clients[-1])
        super().aggregate(scaffold_clients)
