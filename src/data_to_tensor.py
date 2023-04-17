import torch
import pandas
import numpy
import matplotlib.pyplot as plot
import glob


if __name__ == "__main__":

    csv_files = glob.glob("data/*.csv")
    assert len(csv_files) > 0
    src_len = 150
    tgt_len = 50
    L = src_len + tgt_len

    start_index = 0
    shift = 50
    pivot_index = src_len - 1

    collected_tensors = []

    for csv in csv_files:
        print(f"processing {csv}")
        df = pandas.read_csv(csv)
        while start_index + L < df.shape[0]:
            d = df.iloc[start_index : start_index + L, 1:]
            x = df.iloc[start_index : start_index + L, 1:]

            # scaling data
            for column_index in range(d.shape[1]):
                arr = numpy.array(d.iloc[:, column_index])
                pivot_value = arr[pivot_index]
                d.iloc[:, column_index] = (arr - pivot_value) / pivot_value
                pass

            t = torch.tensor(numpy.array(d)).unsqueeze(0)
            collected_tensors.append(t)
            start_index += shift

    out_tensor = torch.cat(collected_tensors, dim=0)
    out_tensor = out_tensor.permute(0, 2, 1)
    print (f"out_tensor shape = {out_tensor.shape}")
    out_tensor = out_tensor.type(torch.FloatTensor)
    torch.save(out_tensor, "data.pt")

    sample_index = 0
    plot.plot(out_tensor[sample_index, 0, :], 'g.')
    plot.plot(out_tensor[sample_index, 0, :], 'g')
    plot.grid()
    plot.show()





