from computing.core.module import Module
from computing.core.dtype import LearningVariable
from compute_platform.computing.base_strategy.modules.network.base_model.linear import (
    MLP,
)

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax, to_dense_adj

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchsde import sdeint


class EgraphData(Data):
    def __init__(self, entity_relation_mat=None):
        super(EgraphData, self).__init__()
        self.x = torch.tensor([])  # [n, 768]
        self.y = torch.tensor([])  # [n, ]
        self.edge_index = torch.tensor([])  # [2, ne]

        self.x_pos = torch.tensor([])  # [n, ] relative position in one earning call
        self.x_time = torch.tensor([])  # [n, ] time for a node
        self.x_type = torch.tensor([])  # [n, ] ec=0, topic=1, entity=2, stock=3
        self.edge_type = torch.tensor(
            []
        )  # [ne, ] ec-topic(01)=0, ec-entity(02)=1, entity-entity(22)=2, stock-ec(03)=3

        self.entity_relation_mat = entity_relation_mat
        self.topic_dim = 768

    def add_node(self, x: torch.tensor, y: int, x_type: int, x_time: float, x_pos=-1):
        # step 1 search for existing node and update time
        if x_pos == -1:
            flag = self.search_and_update_topic_entity_node(y, x_type, x_time)
            if flag:
                return
        # step 2 add new node to graph
        if x.shape[1] != 768:
            self.topic_dim = x.shape[1]
            x = F.pad(x, (0, self.x.shape[1] - x.shape[1]))

        self.x = torch.concat([self.x, x])
        self.y = torch.concat([self.y, torch.tensor([y])])

        self.x_pos = torch.concat([self.x_pos, torch.tensor([x_pos])])
        self.x_time = torch.concat([self.x_time, torch.tensor([x_time])])
        self.x_type = torch.concat([self.x_type, torch.tensor([x_type])])

        # step 3 add entity-entity relation
        if x_type == 2:
            self.update_entity_relation(y)

    def add_edge(
        self,
        source_x_type,
        source_x_pos,
        source_y,
        target_x_type,
        target_x_pos,
        target_y,
        source_x_time=None,
        target_x_time=None,
    ):
        if source_x_type != 0:
            source_index = (
                (self.y == source_y)
                & (self.x_type == source_x_type)
                & (self.x_pos == source_x_pos)
            ).nonzero()
        else:
            assert source_x_time is not None
            source_index = (
                (self.y == source_y)
                & (self.x_type == source_x_type)
                & (self.x_pos == source_x_pos)
                & (self.x_time == source_x_time)
            ).nonzero()
        if target_x_type != 0:
            target_index = (
                (self.y == target_y)
                & (self.x_type == target_x_type)
                & (self.x_pos == target_x_pos)
            ).nonzero()
        else:
            assert target_x_time is not None
            target_index = (
                (self.y == target_y)
                & (self.x_type == target_x_type)
                & (self.x_pos == target_x_pos)
                & (self.x_time == target_x_time)
            ).nonzero()

        if source_index.shape[0] == 0 or target_index.shape[0] == 0:
            print("Node not exist!")
            return

        assert source_index.shape[0] == 1
        assert target_index.shape[0] == 1

        source_index = int(source_index[0][0])
        target_index = int(target_index[0][0])

        # check if edge exist
        if self.edge_index.shape[0] > 0:
            if (
                (self.edge_index[0, :] == source_index)
                & (self.edge_index[1, :] == target_index)
            ).sum() > 0:
                # print("edge exist!")
                return
            if (
                (self.edge_index[1, :] == source_index)
                & (self.edge_index[0, :] == target_index)
            ).sum() > 0:
                # print("edge exist!")
                return

        edge = torch.tensor([source_index, target_index]).unsqueeze(1)  # shape[2,1]
        self.edge_index = torch.concat([self.edge_index, edge], dim=1)  # shape [2,ne]

        type = self.get_edge_type(source_x_type, target_x_type)
        self.edge_type = torch.concat([self.edge_type, torch.tensor([type])])

    def search_and_update_topic_entity_node(self, y, x_type, x_time):
        # if node already exist, return True; else False
        exist = (self.y == y) & (self.x_type == x_type)  # shape [n,]
        if int(exist.sum()) == 0:
            return False
        else:  # if exist, update time
            exist_index = int(exist.nonzero()[0][0])
            self.x_time[exist_index] = x_time
            return True

    def update_entity_relation(self, y):
        entity_relation = self.entity_relation_mat[y]  # shape [n, ]
        possible_related_entity = entity_relation.nonzero()  # shape [m, 1]
        for i in range(possible_related_entity.shape[0]):
            related_entity_index = int(possible_related_entity[i][0])
            t = ((self.y == related_entity_index) & (self.x_type == 2)).nonzero()
            if t.shape[0] == 1 and related_entity_index != y:
                self.add_edge(
                    source_x_type=2,
                    source_x_pos=-1,
                    source_y=y,
                    target_x_type=2,
                    target_x_pos=-1,
                    target_y=related_entity_index,
                )

    def remove_obsolete(self, current_time, window_size):
        # step 1 select old nodes
        obs_mask = self.x_time > (
            current_time - window_size
        )  # [n, ] 1 for retained nodes, 0 for obsolete ones
        if obs_mask.sum() == self.x_time.shape[0]:
            return
        self.x_time = torch.masked_select(self.x_time, obs_mask)  # [m, ]
        self.x_pos = torch.masked_select(self.x_pos, obs_mask)
        self.x_type = torch.masked_select(self.x_type, obs_mask)
        self.y = torch.masked_select(self.y, obs_mask)
        self.x = torch.masked_select(
            self.x, obs_mask.unsqueeze(1).expand(self.x.shape)
        ).view(-1, 768)

        print(
            "remove {} obsolete nodes.".format(obs_mask.shape[0] - self.x_type.shape[0])
        )
        # step 2 reindex
        new_to_old = obs_mask.nonzero().squeeze(1)  # [m,]
        old_to_new = {}
        for j in range(new_to_old.shape[0]):
            old_to_new[new_to_old[j].item()] = j

        # step 3 remove edges
        obs_mask_edge = torch.ones(self.edge_type.shape)
        for i in range(self.edge_type.shape[0]):
            e1, e2 = self.edge_index[:, i]
            e1 = e1.item()
            e2 = e2.item()
            if e1 in new_to_old and e2 in new_to_old:
                self.edge_index[:, i] = torch.tensor([old_to_new[e1], old_to_new[e2]])
            else:
                obs_mask_edge[i] = 0

        obs_mask_edge = obs_mask_edge == 1
        self.edge_type = torch.masked_select(self.edge_type, obs_mask_edge)
        self.edge_index = torch.masked_select(
            self.edge_index, obs_mask_edge.unsqueeze(0).expand(self.edge_index.shape)
        ).view(2, -1)

        print(
            "remove {} obsolete edges.".format(
                obs_mask_edge.shape[0] - self.edge_type.shape[0]
            )
        )

    @staticmethod
    def get_edge_type(source_x_type, target_x_type):
        s = source_x_type + target_x_type
        if s == 1:
            return 0
        elif s == 2:
            return 1
        elif s == 3:
            return 3
        elif s == 4:
            return 2
        else:
            return -1
            # print("edge type not found for *{}_{}*".format(source_x_type, target_x_type))


class HeFF(Module):
    def __init__(
        self,
        stock_num,
        module_id=-1,
        input_dim=768,
        input_topic_dim=768,
        lstm_dim=32,
        text_dim=16,
        entity_dim=4,
        topic_dim=4,
        lstm_layer=1,
        hgt_layer=2,
        divide_window=True,
        wo_hgt=False,
        wo_sde=True,
        wo_entity=False,
        wo_topic=False,
        use_wiki_graph=False,
        use_price_graph=False,
        wiki_relation_num=1,
    ):
        super(HeFF, self).__init__(module_id=module_id)
        self.prediction = LearningVariable(None)
        self.register_decide_hooks(["prediction"])
        # ablation parameter
        self.wo_hgt = wo_hgt
        self.wo_sde = wo_sde

        # model parameter
        self.predict_windows = torch.tensor([0.0, 1.0, 3.0, 7.0, 15.0, 30.0])
        self.time_regular = 100
        self.output_dim = (
            lstm_dim + text_dim + entity_dim + topic_dim
            if not wo_hgt and not use_wiki_graph and not use_price_graph
            else lstm_dim
        )
        self.divide_window = divide_window

        # net
        self.dropout = nn.Dropout(0.2)
        self.hgt = SpatialTenporalRelationModule(
            input_dim,
            self.output_dim,
            lstm_dim,
            text_dim,
            entity_dim,
            topic_dim,
            input_topic_dim,
            lstm_layer,
            hgt_layer,
            lstm_dim,
            wo_hgt,
            wo_entity,
            wo_topic,
            use_wiki_graph,
            use_price_graph,
            wiki_relation_num,
        )
        self.sde = SDEnet(in_dim=self.output_dim, output_dim_list=[self.output_dim])
        if divide_window:
            self.class_net = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.output_dim, 1)
                        # nn.ReLU(),
                        # nn.Linear(self.output_dim//4, 2),
                        # , nn.Softmax(dim=-1)
                    )
                    for _ in range(5)
                ]
            )
        else:
            self.class_net = nn.Sequential(nn.Linear(self.output_dim, 1))

    def forward(self, batch_data):
        # ========================= 1-SpatialTemporalRelation module =========================
        hgt_embed = self.hgt(batch_data)  # shape [b, output_dim]

        # ========================= 2-sde =========================
        if self.wo_sde:
            sde_embed = hgt_embed.expand(5, hgt_embed.shape[0], hgt_embed.shape[1])
        else:
            sde_embed = sdeint(
                self.sde,
                y0=hgt_embed,
                ts=self.predict_windows / self.time_regular,
                method="euler",
                adaptive=False,
            )  # shape [6, b, output_dim]
            sde_embed = sde_embed[1:, :, :]  # [5, b, output_dim]

        # ========================= 3-classification net =========================
        if self.divide_window == True:
            pred = torch.concat(
                [self.class_net[i](sde_embed[i, :, :]).unsqueeze(0) for i in range(5)],
                dim=0,
            )  # [5,b,2]
            pred = pred.transpose(0, 1)
        else:
            pred = self.class_net(sde_embed.view(-1, self.output_dim)).view(
                -1, 5, 1
            )  # shape [b, 5, 2]
        self.prediction = self.dropout(pred)


class SpatialTenporalRelationModule(nn.Module):
    def __init__(
        self,
        input_dim,
        graph_dim,
        lstm_dim,
        text_dim,
        entity_dim,
        topic_dim,
        input_topic_dim,
        lstm_layer=1,
        hgt_layer=2,
        price_dim=0,
        wo_hgt=True,
        wo_entity=False,
        wo_topic=False,
        use_wiki_graph=False,
        use_price_graph=False,
        wiki_relation_num=1,
    ):
        super(SpatialTenporalRelationModule, self).__init__()
        self.wo_hgt = wo_hgt
        self.wo_entity = wo_entity
        self.wo_topic = wo_topic
        self.use_wiki_graph = use_wiki_graph
        self.use_price_graph = use_price_graph

        assert lstm_dim % lstm_layer == 0
        self.input_topic_dim = input_topic_dim
        self.graph_dim = lstm_dim if use_wiki_graph or use_price_graph else graph_dim
        self.lstm_dim = lstm_dim
        self.price_dim = price_dim
        self.text_dim = text_dim
        self.entity_dim = entity_dim
        self.topic_dim = topic_dim
        self.hgt_layer = hgt_layer

        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=lstm_dim // lstm_layer,
            num_layers=lstm_layer,
            bidirectional=False,
            batch_first=True,
        )
        self.text_node_init = nn.Linear(768, text_dim)
        self.topic_node_init = nn.Linear(input_topic_dim, topic_dim)
        self.entity_node_init = nn.Linear(768, entity_dim)  # nn.embedding todo
        self.stock_node_init = nn.Linear(lstm_dim, self.price_dim)
        if use_wiki_graph:
            self.hgt_net = nn.ModuleList(
                [
                    HGTConv(
                        in_dim=self.graph_dim,
                        out_dim=self.graph_dim,
                        num_types=1,
                        num_relations=wiki_relation_num,
                        n_heads=2,
                    )
                    for _ in range(hgt_layer)
                ]
            )
        elif use_price_graph:
            self.hgt_net = nn.ModuleList(
                [
                    HGTConv(
                        in_dim=self.graph_dim,
                        out_dim=self.graph_dim,
                        num_types=1,
                        num_relations=1,
                        n_heads=2,
                    )
                    for _ in range(hgt_layer)
                ]
            )
        else:
            self.hgt_net = nn.ModuleList(
                [
                    HGTConv(
                        in_dim=self.graph_dim,
                        out_dim=self.graph_dim,
                        num_types=4,
                        num_relations=4,
                        n_heads=2,
                    )
                    for _ in range(hgt_layer)
                ]
            )

    def forward(self, batch_data):
        # input batch_data,
        # output embedding for current_stock
        if self.wo_hgt:
            # batch_data.series_context["ohlcv"] # shape [b, context_window_size, stock_num, 5]
            stock_ids = batch_data.series_context["stock_id"]  # list(b) of stock_ids
            batch_size = len(stock_ids)
            history_ohlcv = torch.concat(
                [
                    batch_data.series_context["ohlcv"][i : i + 1, :, stock_ids[i], :]
                    for i in range(batch_size)
                ],
                dim=0,
            )  # shape [b, window, 5]
            history_ohlcv = history_ohlcv / history_ohlcv[0, 0, :]

            price_embed = self.lstm(history_ohlcv)[1][
                0
            ]  # shape [num-layers, b, hidden_dim]
            price_embed = price_embed.transpose(0, 1).reshape(
                batch_size, -1
            )  # shape [b,hidden_dim]

            return price_embed  # shape [b,hidden_dim]

        else:
            # batch_data.series_context["graph"]/["audio"] list of pyg/tensor
            stock_embed = torch.zeros(
                (len(batch_data.series_context["stock_id"]), self.graph_dim)
            ).to("cuda")
            for i in range(len(batch_data.series_context["stock_id"])):
                torch.cuda.empty_cache()
                if self.use_wiki_graph:
                    history_ohlcv = batch_data.series_context["ohlcv"][i].transpose(
                        0, 1
                    )  # shape [s, window, 5]
                    history_ohlcv = history_ohlcv / history_ohlcv[0, 0, :]  # normalize
                    node_emb = self.lstm(history_ohlcv)[1][0].squeeze(
                        0
                    )  # shape [stock_num, hidden_dim]
                    node_type = torch.zeros((node_emb.shape[0])).to("cuda")

                    edge_index = (
                        batch_data.external_data[:, :, :-1]
                        .nonzero()
                        .t()
                        .contiguous()
                        .to("cuda")
                    )
                    edge_type = edge_index[2, :]
                    edge_time = torch.zeros((edge_index.shape[1])).int().to("cuda")
                    edge_index = edge_index[:2, :]

                    for k in range(self.hgt_layer):
                        node_emb = self.hgt_net[k](
                            node_inp=node_emb,
                            node_type=node_type,
                            edge_index=edge_index,
                            edge_type=edge_type,
                            edge_time=edge_time,
                            audio=None,
                        )  # shape [num_nodes, graph_dim]

                    one_stock_emb = node_emb[
                        batch_data.series_context["stock_id"][i]
                    ]  # [graph_dim,]

                elif self.use_price_graph:
                    history_ohlcv = batch_data.series_context["ohlcv"][i].transpose(
                        0, 1
                    )  # shape [s, window, 5]
                    history_ohlcv = history_ohlcv / history_ohlcv[0, 0, :]  # normalize
                    node_emb = self.lstm(history_ohlcv)[1][0].squeeze(
                        0
                    )  # shape [stock_num, hidden_dim]
                    node_type = torch.zeros((node_emb.shape[0])).to("cuda")

                    cov_mat = torch.cov(
                        batch_data.series_context["history_price"][i].transpose(0, 1)
                    )
                    cov_mat = cov_mat - torch.diag_embed(
                        torch.diag(cov_mat)
                    )  # 对角元素置为0
                    cov_mat = cov_mat > 0
                    edge_index = cov_mat.nonzero().t().contiguous()

                    edge_type = torch.zeros((edge_index.shape[1])).to("cuda")
                    edge_time = torch.zeros((edge_index.shape[1])).int().to("cuda")

                    for k in range(self.hgt_layer):
                        node_emb = self.hgt_net[k](
                            node_inp=node_emb,
                            node_type=node_type,
                            edge_index=edge_index,
                            edge_type=edge_type,
                            edge_time=edge_time,
                            audio=None,
                        )  # shape [num_nodes, graph_dim]

                    one_stock_emb = node_emb[
                        batch_data.series_context["stock_id"][i]
                    ]  # [graph_dim,]

                else:
                    # ========================= step 1 lstm for stock embed and add stock_node =========================
                    graph = self.graph_process_add_stock(
                        batch_data.series_context["ohlcv"][i],  # [window, s, 5]
                        batch_data.series_context["earning_call_graph"][
                            i
                        ],  # egraphdata
                    )

                    # ========================= step 2 hgt net =========================
                    (
                        node_emb,
                        node_type,
                        node_pos,
                        node_time,
                        biedge_index,
                        edge_type,
                        edge_time,
                    ) = self.graph_extract_for_hgt(
                        graph, i
                    )  # node_emb.shape [b.768]
                    # generate H0 embedding, align to the same dimension
                    node_emb = self.graph_generate_original_embed(
                        node_emb, node_type
                    )  # shape [b, graph_dim]
                    audio_emb = self.map_audio_to_edge(
                        batch_data.series_context["earning_call_audio"][i],
                        node_type,
                        node_pos,
                        graph.y.to("cuda"),
                        batch_data.series_context["stock_id"][i],
                        edge_type,
                        biedge_index,
                        edge_time,
                    )  # shape [biedge_num, 29]
                    torch.cuda.empty_cache()
                    del node_pos, node_time

                    for k in range(self.hgt_layer):
                        node_emb = self.hgt_net[k](
                            node_inp=node_emb,
                            node_type=node_type,
                            edge_index=biedge_index,
                            edge_type=edge_type,
                            edge_time=edge_time,
                            audio=audio_emb,
                        )  # shape [num_nodes, 768]

                    # ========================= step 3 select stock node embed =========================
                    one_stock_emb = node_emb[
                        (
                            (graph.x_type == 3)
                            & (graph.y == batch_data.series_context["stock_id"][i])
                        )
                    ]  # [1, graph_dim]
                    assert one_stock_emb.shape[0] == 1  # 找到唯一的stock node
                    del node_emb, node_type, biedge_index, edge_type, edge_time

                stock_embed[i, :] = one_stock_emb

            return stock_embed

    def graph_process_add_stock(self, price, graph):
        # 1-select stock node in current graph
        cur_stock_id = graph.y[graph.x_type == 0].unique()
        history_ohlcv = torch.concat(
            [price[:, int(k) : int(k) + 1, :] for k in cur_stock_id], dim=1
        )  # shape [window, stock_num, 5]

        # 2-generate original embedding for each stock node
        history_ohlcv = history_ohlcv.transpose(0, 1)  # shape [s, window, 5]
        history_ohlcv = history_ohlcv / history_ohlcv[0, 0, :]  # normalize
        stock_embed = torch.zeros((len(cur_stock_id), 768)).to("cuda")
        lstm_output = self.lstm(history_ohlcv)[1][
            0
        ]  # shape [num-layers, stock_num, hidden_dim]
        stock_embed[:, : self.lstm_dim] = lstm_output.transpose(0, 1).view(
            len(cur_stock_id), -1
        )  # shape [stock_num, lstm_dim]

        # 3-add stock nodes and edges to graph
        for i in range(len(cur_stock_id)):
            x_embed = stock_embed[i : i + 1, :]  # [1, hidden_dim * num-layers]
            stock_index = cur_stock_id[i]
            # stock node.time = 这个stock的最新的ec的时间
            correspond_ec_nodes_index = (graph.x_type == 0) & (graph.y == stock_index)
            time = graph.x_time[correspond_ec_nodes_index].max()
            correspond_ec_nodes_index = correspond_ec_nodes_index.nonzero().squeeze(
                dim=1
            )

            graph.add_node(x=x_embed.to("cpu"), y=stock_index, x_type=3, x_time=time)
            # ========================= add edge =========================
            new_edge_index = torch.ones((2, correspond_ec_nodes_index.shape[0])) * (
                graph.x.shape[0] - 1
            )
            new_edge_index[1, :] = correspond_ec_nodes_index
            new_edge_type = torch.ones(correspond_ec_nodes_index.shape[0]) * 3

            graph.edge_index = torch.concat([graph.edge_index, new_edge_index], dim=1)
            graph.edge_type = torch.concat([graph.edge_type, new_edge_type])

        return graph

    def graph_extract_for_hgt(self, graph, i):
        if self.wo_entity:
            edge_pos = (graph.edge_type != 1) & (graph.edge_type != 2)
            graph.edge_index = graph.edge_index[:, edge_pos]
            graph.edge_type = graph.edge_type[edge_pos]
        if self.wo_topic:
            edge_pos = graph.edge_type != 0
            graph.edge_index = graph.edge_index[:, edge_pos]
            graph.edge_type = graph.edge_type[edge_pos]
        # 1-Generate an undirected edge index
        biedge_index = torch.concat(
            [graph.edge_index.long()[1:2, :], graph.edge_index.long()[0:1, :]], dim=0
        )  # [2, enum]
        biedge_index = torch.concat(
            [graph.edge_index.long(), biedge_index], dim=1
        )  # [2, 2*enum]

        # 2-edgetype
        biedge_type = torch.concat([graph.edge_type, graph.edge_type])

        # 3-edge time
        time_0 = torch.index_select(graph.x_time, 0, biedge_index[0, :])  # [2enum,]
        time_1 = torch.index_select(graph.x_time, 0, biedge_index[1, :])  # [2enum,]
        edge_time = abs(time_0 - time_1).int()

        return (
            graph.x.to("cuda"),
            graph.x_type.to("cuda"),
            graph.x_pos.to("cuda"),
            graph.x_time.to("cuda"),
            biedge_index.to("cuda"),
            biedge_type.to("cuda"),
            edge_time.to("cuda"),
        )

    def graph_generate_original_embed(self, node_emb, node_type):
        res_node_emb = torch.zeros((node_emb.shape[0], self.graph_dim)).to(
            "cuda"
        )  # [stock, text, topic, entity]
        for i in range(4):
            idx = node_type == i
            if i == 0:  # text node
                res_node_emb[idx, self.price_dim : (self.price_dim + self.text_dim)] = (
                    self.text_node_init(node_emb[idx])
                )
            elif i == 1:  # topic node
                res_node_emb[
                    idx,
                    (self.price_dim + self.text_dim) : (
                        self.price_dim + self.text_dim + self.topic_dim
                    ),
                ] = self.topic_node_init(node_emb[idx, : self.input_topic_dim])
            elif i == 2:  # entity node
                res_node_emb[
                    idx, (self.price_dim + self.text_dim + self.topic_dim) :
                ] = self.entity_node_init(node_emb[idx])
            elif i == 3:  # stock node
                res_node_emb[idx, : self.price_dim] = self.stock_node_init(
                    node_emb[idx, : self.lstm_dim]
                )

        return res_node_emb

    def map_audio_to_edge(
        self,
        audio_embed,
        node_type,
        node_pos,
        node_y,
        stock_id,
        edge_type,
        biedge_index,
        edge_time,
    ):
        res_audio = torch.zeros((edge_type.shape[0], 29)).to("cuda")
        type_target = torch.index_select(node_type, 0, biedge_index[1, :])  # [2enum,]
        type_source = torch.index_select(node_type, 0, biedge_index[0, :])  # [2enum,]
        y_source = torch.index_select(node_y, 0, biedge_index[0, :])

        selected_edge_index = (
            (edge_type == 3)
            & (edge_time == 0)
            & (type_target == 3)
            & (type_source == 0)
            & (y_source == stock_id)
        )
        text_pos = torch.index_select(node_pos, 0, biedge_index[0, :])
        audio_embed = audio_embed / audio_embed.mean(dim=0)
        for i in range(audio_embed.shape[0]):
            t = (text_pos == i) & selected_edge_index
            res_audio[t] = audio_embed[i]

        return res_audio


class RelTemporalEncoding(nn.Module):
    """
    Implement the Temporal Encoding (Sinusoid) function.
    """

    def __init__(self, n_hid, max_len=240, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class HGTConv(MessagePassing):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout=0.2,
        use_norm=True,
        use_RTE=True,
        **kwargs
    ):
        super(HGTConv, self).__init__(node_dim=0, aggr="add", **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_RTE = use_RTE
        self.att = None

        self.k_linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for _ in range(num_types)]
        )
        self.q_linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for _ in range(num_types)]
        )
        self.v_linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for _ in range(num_types)]
        )
        self.a_linears = nn.ModuleList(
            [nn.Linear(out_dim, out_dim) for _ in range(num_types)]
        )
        self.norms = (
            nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(num_types)])
            if use_norm
            else nn.ModuleList()
        )
        self.text_audio_merge_net = nn.ModuleList(
            [nn.Linear(in_dim, out_dim), nn.Linear(29, in_dim)]
        )

        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        if self.use_RTE:
            self.emb = RelTemporalEncoding(in_dim)

        glorot(self.relation_att)
        glorot(self.relation_msg)

    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time, audio):
        return self.propagate(
            edge_index,
            node_inp=node_inp,
            node_type=node_type,
            edge_type=edge_type,
            edge_time=edge_time,
            audio=audio,
        )

    def message(
        self,
        edge_index_i,
        node_inp_i,
        node_inp_j,
        node_type_i,
        node_type_j,
        edge_type,
        edge_time,
        audio=None,
    ):
        """
        j: source, i: target; <j, i>
        """
        data_size = edge_index_i.size(0)  # edge_num
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        for source_type in range(self.num_types):
            sb = node_type_j == int(source_type)
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]

            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]

                for relation_type in range(self.num_relations):
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    # ========================= Step 1: Heterogeneous Mutual Attention =========================
                    if source_type == 0 and target_type == 3 and audio is not None:
                        audio_node_pos = audio[idx].sum(dim=1) != 0
                        source_node_vec = source_node_vec[audio_node_pos]
                        source_node_vec_with_audio = self.text_audio_merge_net[0](
                            source_node_vec
                        ) + self.text_audio_merge_net[1](audio[idx][audio_node_pos])
                        q_mat = q_linear(source_node_vec_with_audio).view(
                            -1, self.n_heads, self.d_k
                        )  # self.attention
                        k_mat = k_linear(source_node_vec_with_audio).view(
                            -1, self.n_heads, self.d_k
                        )
                        idx = idx & (audio.sum(dim=1) != 0)
                    else:
                        q_mat = q_linear(target_node_vec).view(
                            -1, self.n_heads, self.d_k
                        )
                        k_mat = k_linear(source_node_vec).view(
                            -1, self.n_heads, self.d_k
                        )
                    k_mat = torch.bmm(
                        k_mat.transpose(1, 0), self.relation_att[relation_type]
                    ).transpose(1, 0)
                    res_att[idx] = (
                        (q_mat * k_mat).sum(dim=-1)
                        * self.relation_pri[relation_type]
                        / self.sqrt_dk
                    )  # eq 2 Attention
                    # ========================= Step 2: Heterogeneous Message Passing =========================
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(
                        v_mat.transpose(1, 0), self.relation_msg[relation_type]
                    ).transpose(1, 0)

        # Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        self.att = softmax(res_att, edge_index_i)  # shape [num_edge, num_heads]
        res = res_msg * self.att.view(
            -1, self.n_heads, 1
        )  # res_msg.shape [num_edge, num_head, graph_dim/num_head]
        del res_att, res_msg
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        """
        Step 3: Target-specific Aggregation
        x = W[node_type] * gelu(Agg(x)) + x
        """
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = node_type == int(target_type)
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            """
                Add skip connection with learnable weight self.skip[t_id]
            """
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](
                    trans_out * alpha + node_inp[idx] * (1 - alpha)
                )
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return "{}(in_dim={}, out_dim={}, num_types={}, num_types={})".format(
            self.__class__.__name__,
            self.in_dim,
            self.out_dim,
            self.num_types,
            self.num_relations,
        )


class SDEnet(nn.Module):
    def __init__(
        self, in_dim, output_dim_list=None, activation=None, last_activation=None
    ):
        super(SDEnet, self).__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        if output_dim_list is None:
            output_dim_list = [in_dim]
        else:
            output_dim_list.append(in_dim)

        self.f_net = MLP(in_dim, output_dim_list, activation, last_activation)
        self.g_net = MLP(in_dim, output_dim_list, activation, last_activation)

    def f(self, t, y):
        return self.f_net(y)

    def g(self, t, y):
        return self.g_net(y)
