import torch as torch
import torch.nn.functional as F


from computing.core.container import Container
from computing.core.dtype import ContainerProperty
from computing.utils.data_struct import ContextSeriesTensorData
from computing.base_strategy.containers.prediction.classification.predictor_movement import (
    PredictorMovement,
)


__all__ = ["EarningCallFilter", "EarningCallPrepocessor"]


def calculate_volatility(return_price):
    """
    Daily RETURN for future periods
    """
    if len(return_price.shape) == 2:  # single_day
        return return_price.abs().log()
    elif return_price.shape[2] == 1:  # single_day
        return return_price.squeeze(2).abs().log()
    else:  # multi_day
        return_price_avg = return_price.mean(-1, keepdim=True)
        return (return_price - return_price_avg).pow(2).mean(-1).sqrt().log()


class EarningCallFilter(Container):
    def __init__(
        self,
        container_id=-1,
        batch_size=64,
        predict_window_size=3,
        stored_key_list=None,
    ):
        super(EarningCallFilter, self).__init__(container_id=container_id)
        self.batch_size = ContainerProperty(batch_size)
        self.predict_window_size = ContainerProperty(predict_window_size)

        # buffer
        self.stored_key_list = stored_key_list
        self.buffer = None
        self.is_train = False

    def init_empty_tensor_data(self, formate_refer_data):
        empty_data = ContextSeriesTensorData(
            index_map_dict={
                key: {} for key in formate_refer_data.index_map_dict.keys()
            },
            series_context={
                key: [] for key in formate_refer_data.series_context.keys()
            },
            environment_context={},
            reward=torch.tensor([]).to(self.device),
            window_size=formate_refer_data.window_size,
            context_window_size=formate_refer_data.context_window_size,
            packed=formate_refer_data.packed,
        )
        empty_data.series_context["stock_id"] = []
        empty_data.series_context["date"] = []
        return empty_data

    def get_batch_data(self):
        batch_data = self.init_empty_tensor_data(self.buffer)
        for key, value in self.buffer.series_context.items():
            batch_data.series_context[key] = self.buffer.series_context[key][
                : self.batch_size
            ]
            self.buffer.series_context[key] = self.buffer.series_context[key][
                self.batch_size :
            ]

        batch_data.reward = self.buffer.reward[: self.batch_size]
        self.buffer.reward = self.buffer.reward[self.batch_size :]
        return batch_data

    def store_data_to_buffer(self, batch_data, **kwargs):  # 返回has_call
        # step1: 根据 batch_data 中 earning_call 是否为 None 设置 mask
        if "earning_call_file_name" in batch_data.series_context.keys():
            earning_call_info = batch_data.series_context["earning_call_file_name"]
            batch_size = len(earning_call_info)
            stock_num = len(earning_call_info[0])
        else:
            earning_call_audio = batch_data.series_context["earning_call_audio"]
            earning_call_text = batch_data.series_context["earning_call_text"]
            batch_size = len(earning_call_text)
            stock_num = len(earning_call_text[0])

        ground_truth = calculate_volatility(
            batch_data.reward[..., : self.predict_window_size] - 1
        )
        if self.stored_key_list is None:
            self.stored_key_list = list(batch_data.series_context.keys())

        for b in range(batch_size):
            for s in range(stock_num):
                # 判断波动率是否存在异常值
                is_not_anomaly = ground_truth[b][s] > -23  # log(1e-10)
                # exist_audio_text = (earning_call_audio[b][s][-1] is not None) and (earning_call_text[b][s][-1] is not None)
                exist_audio_text = (
                    earning_call_info[b][s][-1] is not None
                )  # todo 仅限price only算法

                if exist_audio_text and is_not_anomaly:
                    # if earning_call_text[b][s][-1].shape[0] <= 1:  # earning call只有1句话时预测值为nan
                    #     continue
                    if "earning_call_word" in batch_data.series_context.keys():
                        if (
                            batch_data.series_context["earning_call_word"][b][s][-1]
                            is None
                        ):
                            print("no words pyg data!")
                            continue
                    if self.is_train is False:
                        previous = calculate_volatility(
                            batch_data.series_context["history_price"][
                                :, -self.predict_window_size :
                            ].transpose(1, 2)
                            - 1
                        )
                        if previous[b][s] <= -23:
                            continue

                    # step2: 将series_context数据写入buffer
                    if self.buffer.reward is None:
                        self.buffer.reward = batch_data.reward[b : b + 1, s : s + 1]
                    else:
                        self.buffer.reward = torch.cat(
                            [
                                self.buffer.reward,
                                batch_data.reward[b : b + 1, s : s + 1],
                            ],
                            dim=0,
                        )

                    self.buffer.series_context["stock_id"].append(
                        batch_data.index_map_dict["series_index_map"][s]
                    )
                    self.buffer.series_context["date"].append(
                        batch_data.index_map_dict["round_index_map"][
                            batch_data.init_index - 1
                        ]
                    )
                    for stored_key in self.stored_key_list:
                        if stored_key in ["history_price", "ohlcv"]:
                            if len(self.buffer.series_context[stored_key]) == 0:
                                self.buffer.series_context[stored_key] = (
                                    batch_data.series_context[stored_key][
                                        b : b + 1, :, s : s + 1
                                    ]
                                )
                            else:
                                self.buffer.series_context[stored_key] = torch.cat(
                                    [
                                        self.buffer.series_context[stored_key],
                                        batch_data.series_context[stored_key][
                                            b : b + 1, :, s : s + 1
                                        ],
                                    ],
                                    dim=0,
                                )
                        else:
                            self.buffer.series_context[stored_key].append(
                                [batch_data.series_context[stored_key][b][s][-1]]
                            )  # mod for news

    def train(self, batch_data: ContextSeriesTensorData, **kwargs):
        # batch_data.series_context ['history_price'] shape [batch, window_size, stock_num] e.g. [32, 60, 241]
        # batch_data.series_context ['ohlcv'] shape [batch, window_size, stock_num, feature_num] e.g. [32, 60, 241, 5]
        # batch_data.series_context ['earning_call_...'] list length of batch_size, comprised of lists length of stock_num (e.g. 241
        # batch_data.series_context ['result'] list length of batch_size, comprised of lists length of stock_num
        # 踩坑：dict.fromkeys(, [])会出现浅拷贝
        # assert "earning_call_text" in batch_data.series_context.keys()
        # assert "earning_call_file_name" in batch_data.series_context.keys()

        # step 1: init
        if self.is_train is False:
            self.is_train = True
            self.buffer = self.init_empty_tensor_data(batch_data)

            # step2: 将的batch_data筛选并写入buffer
        self.store_data_to_buffer(batch_data, **kwargs)

        # step3: 若buffer长度大于batch_size, 返回对应的batch
        train_data_list = []
        while (
            len(self.buffer.series_context["earning_call_file_name"]) >= self.batch_size
        ):
            train_data_list.append(self.get_batch_data())

        return {"train_data_list": train_data_list}

    def decide(self, data: ContextSeriesTensorData, **kwargs):
        self.is_train = False
        self.buffer = self.init_empty_tensor_data(data)

        self.store_data_to_buffer(data, **kwargs)
        self.buffer.index_map_dict["round_index_map"] = data.index_map_dict[
            "round_index_map"
        ]
        if len(self.buffer.series_context["earning_call_file_name"]) != 0:
            return {"test_data": self.buffer}
        else:
            return {"test_data": None}


class EarningCallFilterEGraph(EarningCallFilter):
    def __init__(
        self,
        container_id=-1,
        batch_size=64,
        predict_window_size=3,
        stored_key_list=None,
    ):
        super(EarningCallFilterEGraph, self).__init__(
            container_id=container_id,
            batch_size=batch_size,
            predict_window_size=predict_window_size,
            stored_key_list=stored_key_list,
        )

    def store_data_to_buffer(self, batch_data, **kwargs):
        # step1: 根据 batch_data 中 earning_call 是否为 None 设置 mask
        earning_call_audio = batch_data.series_context["earning_call_audio"]
        ground_truth = calculate_volatility(
            batch_data.reward[..., : self.predict_window_size] - 1
        )

        batch_size = len(earning_call_audio)
        stock_num = len(earning_call_audio[0])

        if self.stored_key_list is None:
            self.stored_key_list = list(batch_data.series_context.keys())

        for b in range(batch_size):
            for s in range(stock_num):
                # 判断波动率是否存在异常值
                is_not_anomaly = ground_truth[b][s] > -23  # log(1e-10)
                if earning_call_audio[b][s][-1] is not None and is_not_anomaly:
                    if "earning_call_graph" in batch_data.series_context.keys():
                        if (
                            batch_data.series_context["earning_call_graph"][b][s][-1]
                            is None
                        ):
                            continue
                    if (
                        len(earning_call_audio[b][s][-1].shape) == 1
                        or earning_call_audio[b][s][-1].shape[0] < 5
                    ):
                        continue
                    if self.is_train is False:
                        previous = calculate_volatility(
                            batch_data.series_context["history_price"][
                                :, -self.predict_window_size :
                            ].transpose(1, 2)
                            - 1
                        )
                        if previous[b][s] <= -23:
                            continue
                    # step2: Write series_context data to buffer
                    if self.buffer.reward is None:
                        self.buffer.reward = batch_data.reward[b : b + 1, s : s + 1]
                    else:
                        self.buffer.reward = torch.cat(
                            [
                                self.buffer.reward,
                                batch_data.reward[b : b + 1, s : s + 1],
                            ],
                            dim=0,
                        )

                    self.buffer.series_context["stock_id"].append(s)
                    self.buffer.series_context["date"].append(
                        batch_data.index_map_dict["round_index_map"][
                            batch_data.init_index - 1
                        ]
                    )
                    for stored_key in self.stored_key_list:
                        if stored_key in ["history_price", "ohlcv"]:
                            if len(self.buffer.series_context[stored_key]) == 0:
                                self.buffer.series_context[stored_key] = (
                                    batch_data.series_context[stored_key][
                                        b : b + 1, :, :
                                    ]
                                )
                            else:
                                self.buffer.series_context[stored_key] = torch.cat(
                                    [
                                        self.buffer.series_context[stored_key],
                                        batch_data.series_context[stored_key][
                                            b : b + 1, :, :
                                        ],
                                    ],
                                    dim=0,
                                )
                        else:
                            self.buffer.series_context[stored_key].append(
                                batch_data.series_context[stored_key][b][s][-1]
                            )  # mod for news


class EarningCallPrepocessor(Container):
    def __init__(
        self,
        container_id=-1,
        max_seq_len=None,
        feature_dims=None,
        is_chunk_padding=True,
    ):
        super(EarningCallPrepocessor, self).__init__(container_id=container_id)
        self.max_seq_len = max_seq_len
        self.feature_dims = feature_dims
        self.is_chunk_padding = is_chunk_padding

    @staticmethod
    def chunk_padding_earning_call(data, text_dim, audio_dim, device, max_seq_len=None):
        def chunk_padding_series(series, feature_dim, device, max_seq_len=None):
            batch_size = len(series)
            stock_num = len(series[0])

            if max_seq_len is None:
                max_seq_len = 0
                for b in range(batch_size):
                    for s in range(stock_num):
                        if series[b][s] is not None:
                            max_seq_len = max(max_seq_len, len(series[b][s]))

            context = torch.zeros(batch_size, stock_num, max_seq_len, feature_dim).to(
                device
            )
            mask = torch.zeros(batch_size, stock_num, max_seq_len).to(device)
            has_call = torch.zeros(batch_size, stock_num).to(device)
            for b in range(batch_size):
                for s in range(stock_num):
                    if series[b][s] is not None:
                        sentence_num = min(max_seq_len, series[b][s].shape[0])
                        context[b, s, :sentence_num] = series[b][s][:sentence_num]
                        mask[b, s, :sentence_num] = 1
                        has_call[b, s] = 1
            return context, has_call, mask, max_seq_len

        text, has_call, mask, max_seq_len = chunk_padding_series(
            data.series_context["earning_call_text"], text_dim, device, max_seq_len
        )
        audio, _, _, _ = chunk_padding_series(
            data.series_context["earning_call_audio"], audio_dim, device, max_seq_len
        )
        return text, audio, has_call, mask

    @staticmethod
    def normalization_close_price(data):
        ohlc = data.series_context["ohlcv"][..., :4][..., 3]  # [b,w,s]
        ohlc = ohlc / ohlc[:, 0:1, :]  # [b,w,s]
        price = ohlc.permute(0, 2, 1)  # [b,s,w]
        return price

    def preprocess(self, data):
        """
        mask: padding mask
        """
        context = {"price": self.normalization_close_price(data)}
        if self.is_chunk_padding:
            context["text"], context["audio"], context["has_call"], context["mask"] = (
                self.chunk_padding_earning_call(
                    data=data,
                    max_seq_len=self.max_seq_len,
                    text_dim=self.feature_dims["earning_call_text"],
                    audio_dim=self.feature_dims["earning_call_audio"],
                    device=self.device,
                )
            )
        else:
            context["text"], context["audio"] = (
                data.series_context["earning_call_text"],
                data.series_context["earning_call_audio"],
            )
            if "earning_call_word" in data.series_context.keys():
                context["word"] = data.series_context["earning_call_word"]
        return context


class MovementPredictorHeFF(PredictorMovement):
    def __init__(
        self,
        container_id=-1,
        modules_dict={"net": None, "filter": None, "preprocessor": None},
        class_num=2,
        predict_window_size=3,
        accumulation_steps=1,
        lr=1e-3,
        weight_decay=0,
        gamma=1,
        trading=True,
        **kwargs
    ):
        super(MovementPredictorHeFF, self).__init__(
            container_id=container_id,
            modules_dict=modules_dict,
            class_num=class_num,
            predict_window_size=predict_window_size,
            accumulation_steps=accumulation_steps,
            lr=lr,
            weight_decay=weight_decay,
            gamma=gamma,
            **kwargs
        )
        self.predict_periods = [1, 3, 7, 15, 30]
        self.mse = torch.nn.MSELoss(reduction="none")
        # ========================= trading simulation =========================
        self.trading = trading
        if self.trading:
            self.trade_len = 45  # 49
            self.cost = torch.zeros(len(self.predict_periods), 1)
            self.net_profit = [
                torch.zeros(len(self.predict_periods), 1) for _ in range(self.trade_len)
            ]
            self.net_return = [
                torch.zeros(len(self.predict_periods), 1) for _ in range(self.trade_len)
            ]
            self.trade_count = [
                torch.zeros(len(self.predict_periods), 1) for _ in range(self.trade_len)
            ]
            self.init_trade_date = None

    def init_trading(self, predict_periods):
        init_net_profit = torch.zeros(len(predict_periods), 1)
        init_net_return = torch.zeros(len(predict_periods), 1)
        init_cost = torch.zeros(len(predict_periods), 1)
        return init_net_profit, init_net_return, init_cost

    def get_ground_truth(self, reward, **kwargs):
        ground_truth_list = []
        ground_truth_return = []
        for predict_window_size in self.predict_periods:
            ground_truth = PredictorMovement._get_ground_truth(
                reward, predict_window_size=predict_window_size
            )
            ground_truth_list.append(ground_truth)
            ground_truth_return.append(
                reward[..., :predict_window_size].cumprod(-1)[..., -1]
            )

        ground_truth_movement = torch.stack(ground_truth_list, dim=1).squeeze(
            -1
        )  # [batch_size, gt_num]
        ground_truth_return = torch.stack(ground_truth_return, dim=1).squeeze(
            -1
        )  # [batch_size, gt_num]

        return ground_truth_return - 1  # ground_truth_movement

    def get_input_data(self, data, is_train, **kwargs):
        data.external_data = kwargs["external_data"]
        return data

    def decide_(self, data: ContextSeriesTensorData, **kwargs):
        input_data = self.get_input_data(data, is_train=False, **kwargs)
        ground_truth = self.get_ground_truth(data.reward, **kwargs)  # [b,5]

        with torch.no_grad():
            prediction = self.predict(
                input_data, is_train=False, tensor_data=data, **kwargs
            )  # [b,5,1]

        prediction = prediction.squeeze(dim=2)
        mov = torch.zeros((prediction.shape[0], prediction.shape[1], 2))
        mov[:, :, 0] = prediction < 0
        mov[:, :, 1] = prediction >= 0

        result = {"prediction": mov, "ground_truth": (ground_truth >= 0) * 1}

        if self.trading:
            for i in range(mov.shape[0]):
                movement = mov[i, :5, 0]
                self.trade(
                    movement,
                    data.reward[i, : self.predict_periods[-1]].squeeze().to("cpu"),
                    data.series_context["ohlcv"][i, -1, 0, 3].to("cpu"),
                    data.series_context["date"][i],
                    data.index_map_dict["round_index_map"],
                )

        return result

    def trade(self, movement, future_price, close_price, date, date_series):
        if self.init_trade_date is None:
            self.init_trade_date = date
        pos = date_series.index(date) - date_series.index(self.init_trade_date)
        # print(pos)
        for i, period in enumerate(self.predict_periods):
            accumulate_reward = torch.cumprod(future_price[:period], dim=0)
            if movement[i] == 1:  # up
                for j in range(period):
                    if pos + j > self.trade_len - 1:
                        break
                    self.net_return[pos + j][i] += future_price[j] - 1
                    self.trade_count[pos + j][i] += 1
                self.net_profit[pos][i] += (accumulate_reward[-1] - 1) * close_price
            else:
                for j in range(period):
                    if pos + j > self.trade_len - 1:
                        break
                    self.net_return[pos + j][i] += 1 - future_price[j]
                    self.trade_count[pos + j][i] += 1
                self.net_profit[pos][i] += (1 - accumulate_reward[-1]) * close_price
            self.cost[i, 0] += close_price

    def update(self, result: dict, data, **kwargs):
        for i in range(self.trade_len):
            self.trade_count[i][self.trade_count[i] == 0] = 1
        net_profit_avg = [
            self.net_profit[i] / self.trade_count[i] for i in range(self.trade_len)
        ]
        net_return_avg = [
            self.net_return[i] / self.trade_count[i] for i in range(self.trade_len)
        ]

        net_profit_tensor = torch.stack(self.net_profit)
        net_return_tensor = torch.stack(net_return_avg)  # [ec_num, 5, 1]
        cm_profit = torch.sum(net_profit_tensor, dim=0)

        mean_return = torch.mean(net_return_tensor, dim=0)
        std_return = torch.std(net_return_tensor, dim=0)
        sr = mean_return / std_return
        cm_reward = torch.cumprod(net_return_tensor + 1, dim=0)[-1]  # [5,1]

        annual_sr = (
            (pow(mean_return + 1, 255) - 1) / std_return / torch.sqrt(torch.tensor(255))
        )
        for i, period in enumerate(self.predict_periods):
            print(
                "period:{}:profit:{}:std_return:{}:cw:{}:mean_return:{}:sr:{}:annual_sr:{}".format(
                    period,
                    cm_profit[i, 0],
                    std_return[i, 0],
                    cm_reward[i, 0],
                    mean_return[i, 0],
                    sr[i, 0],
                    annual_sr[i, 0],
                )
            )
        return

    def calculate_loss(self, prediction, ground_truth, **kwargs):
        # prediction shape [b, 5, 1]
        # ground_truth shape [b, 5]
        prediction = prediction.squeeze(dim=2)  # shape [b, 5]
        # ========================= cross_entropy loss =========================
        mov = torch.zeros((prediction.shape[0], prediction.shape[1], 2)).to("cuda")
        mov[:, :, 0] = prediction < 0
        mov[:, :, 1] = prediction >= 0
        cross_entropy_loss = self.criterion(
            mov.reshape(-1, 2), ((ground_truth >= 0) * 1).view(-1)
        )
        cross_entropy_loss = cross_entropy_loss.reshape(
            ground_truth.shape
        )  # shape [b,5]
        cross_entropy_loss = torch.sum(cross_entropy_loss, dim=-1)
        cross_entropy_loss = torch.mean(cross_entropy_loss, dim=0)
        # ========================= ranking loss =========================
        # point-wise loss
        point_wise_loss = self.mse(prediction, ground_truth)  # [b, s]
        point_wise_loss = point_wise_loss.sum(dim=-1).mean()  # [b]

        # pair-wise rank-aware loss
        prediction_delta = prediction.unsqueeze(1) - prediction.unsqueeze(
            2
        )  # [b, s, s]
        ground_truth_delta = ground_truth.unsqueeze(1) - ground_truth.unsqueeze(
            2
        )  # [b, s, s]
        pair_wise_loss = (
            F.relu(-(prediction_delta * ground_truth_delta))
            .sum(dim=-1)
            .sum(dim=-1)
            .mean()
        )

        loss = 0.01 * cross_entropy_loss + 1 * point_wise_loss + 0 * pair_wise_loss
        # print("mov_loss:{}, point_wise_loss:{}, pair_wise_loss:{}".format(loss1, point_wise_loss.item(), pair_wise_loss.item()))

        return loss
