import matplotlib
import matplotlib.pyplot as plot
import numpy
import torch
import torch.nn as nn

# import nn_lstm
import nn_lstm_extended

matplotlib.use("Agg")


def plot_function(_source_tensor, pred_values, _index_of_sample):
    L = _source_tensor.shape[2]

    # real test (input part)
    _source_values_numpy = _source_tensor.detach().numpy()

    # plot real past data
    plot.plot(numpy.arange(l_input), _source_values_numpy[_index_of_sample, 0, :l_input], "g", linewidth=2.0)
    plot.plot(numpy.arange(l_input), _source_values_numpy[_index_of_sample, 0, :l_input], ".g", linewidth=2.0)

    # # plot pred past data: # TODO: is it working ?
    # plot.plot(numpy.arange(l_input), pred_values[_index_of_sample, 0, :l_input], "rp:", linewidth=2.0)
    # plot.plot(numpy.arange(l_input), pred_values[_index_of_sample, 0, :l_input], ".r", linewidth=2.0)

    # plot real target (future points)
    plot.plot(numpy.arange(l_input, l_input + future), _source_values_numpy[_index_of_sample, 0, -future:], "g:",
              linewidth=2.0)
    plot.plot(numpy.arange(l_input, l_input + future), _source_values_numpy[_index_of_sample, 0, -future:], "g.",
              linewidth=2.0)

    # plot prediction (future points)
    plot.plot(numpy.arange(l_input, l_input + future), pred_values[_index_of_sample, 0, -future:], "rp:", linewidth=2.0)
    plot.plot(numpy.arange(l_input, l_input + future), pred_values[_index_of_sample, 0, -future:], "rp:", linewidth=2.0)


if __name__ == "__main__":

    T = torch.load("data.pt")       # to create data.pt use data_to_tensor.py
    l_input = 150
    future = T.shape[2] - l_input - 1

    # TODO: move params to separate config file ?
    # Training params ------------------------------------------------
    learning_rate = 0.00025
    n_hidden = 1
    epochs = 10
    batch_size = 1024
    weight_decay = 0.00001
    dropout = 0.01

    # specific for LBFGS optimizer:
    LBFGS_history_size = 100
    LBFGS_max_itr = 100

    num_lstm_stack_layers = 1

    # Prepare data ======================================================
    train_portion = 0.6
    validation_of_test_portion = 0.5

    # [TODO] Scale data if not !

    num_train_size = int(train_portion * T.shape[0])
    num_test_size = T.shape[0] - num_train_size
    num_val_size = int(num_test_size * validation_of_test_portion)
    num_test_size = num_test_size - num_val_size

    print("num_train_size =", num_train_size)
    print("num_val_size =", num_val_size)
    print("num_test_size =", num_test_size)
    print("--------------------------------------------------------------")

    assert num_test_size > 0

    # Split samples into train / validation / test parts
    T_train = T[:num_train_size]
    T_val = T[num_train_size: num_train_size + num_val_size]
    T_test = T[num_train_size + num_val_size:]

    # dim=(batch_size, features, sequence_len)
    train_inputs = T_train[:, :, :l_input]
    train_target = T_train[:, :, 1:]

    val_inputs = T_val[:, :, :l_input]
    val_target = T_val[:, :, 1:]

    test_inputs = T_test[:, :, :l_input]
    test_target = T_test[:, :, 1:]

    # GPU or CPU device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_batches = torch.split(train_inputs, batch_size, dim=0)
    target_batches = torch.split(train_target, batch_size, dim=0)

    val_inputs_batches = torch.split(val_inputs, batch_size, dim=0)
    val_target_batches = torch.split(val_target, batch_size, dim=0)

    test_inputs_batches = torch.split(test_inputs, 1, dim=0)
    test_target_batches = torch.split(test_target, 1, dim=0)

    # Init the model and start training loop:
    # model = nn.LSTMNet(n_hidden=n_hidden,
    #                         device=device,
    #                         num_lstm_stack_layers=num_lstm_stack_layers)

    model = nn_lstm_extended.LSTMExtended(n_hidden=n_hidden,
                                          device=device,
                                          num_lstm_stack_layers=num_lstm_stack_layers,
                                          tgt_future_len=future)

    print(f"number of params = {model.count_params()}")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # optimizer = torch.optim.LBFGS(model.parameters(),
    #                               lr=learning_rate,
    #                               max_iter=LBFGS_max_itr,
    #                               history_size=LBFGS_history_size)

    for i_epoch in range(epochs):
        print(f"epoch {i_epoch}")
        sum_train_loss = 0

        for batch_index, batch in enumerate(train_batches):
            batch = batch.to(device)
            target_batch = target_batches[batch_index].to(device)

            optimizer.zero_grad()
            out = model(batch, future=future)
            loss_ = criterion(out[:, 0:1, :], target_batch[:, 0:1, -future:])
            loss_.backward()

            optimizer.step()
            sum_train_loss += loss_.item()

        print(f"epoch_loss = {sum_train_loss / len(train_batches)}")

        # Validation
        with torch.no_grad():
            val_loss = 0

            for batch_index in range(len(val_inputs_batches)):
                val_input_batch = val_inputs_batches[batch_index].to(device)
                val_target_batch = val_target_batches[batch_index].to(device)

                preds_val = model(val_input_batch, future=future)
                loss = criterion(preds_val[:, 0:1, :], val_target_batch[:, 0:1, -future:])
                val_loss += loss.item()
                y_pred_val = preds_val.cpu().detach().numpy()

                if batch_index == 0:
                    plot.figure(figsize=(20, 8))
                    plot.title(f"Step (Validation set) {i_epoch + 1}")
                    plot.xlabel("x")
                    plot.ylabel("y")

                    plot_function(T_val, y_pred_val, 0)
                    plot.savefig(f"val_plots/predict.{i_epoch}.png")
                    plot.close()

            print("validation loss", val_loss / len(val_inputs_batches))
            print("")

    # Final test set performance:
    model.eval()

    with torch.no_grad():

        test_loss = 0

        for batch_index in range(min(100, len(test_inputs_batches))):
            test_input_batch = test_inputs_batches[batch_index].to(device)
            test_target_batch = test_target_batches[batch_index].to(device)

            preds_test = model(test_input_batch, future=future)
            loss = criterion(preds_test[:, 0:1, :], test_target_batch[:, 0:1, -future:])

            test_loss += loss.item()
            y_pred_test = preds_test.cpu().detach().numpy()

            plot.figure(figsize=(20, 8))
            plot.title(f"Test index {batch_index}")
            plot.xlabel("x")
            plot.ylabel("y")

            plot_function(T_test, y_pred_test, 0)
            plot.savefig(f"test_plots/test.{batch_index}.png")
            plot.close()

        print("Test loss", test_loss / len(test_inputs_batches))
        print("")

    # Saving model for c++
    model.to("cpu")
    model.device = "cpu"
    example_input_to_forward_method = val_inputs[0:1, :, :]
    example_input_to_forward_method = example_input_to_forward_method.to("cpu")
    traced_script_module = torch.jit.trace(model, example_input_to_forward_method)
    traced_script_module.save("lstm.pt")
