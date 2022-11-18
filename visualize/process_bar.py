import sys


# process_bar
def process_bar(process_step, epoch, avg_loss, lr=0, step_loss=0, time_str='', training=True):
    print("\r", end="")
    pad = ' '
    if process_step == 100:
        pad = ''
    if training:
        print(
            "Epoch:\033[34m%d\033[0m Avg loss:\033[34m%.4f\033[0m Step loss:\033[34m%.4f\033[0m lr:\033[34m%.2e\033[0m " % (
                epoch, avg_loss, step_loss, lr) + " " + time_str + "   \033[34mTraining progress: " + pad + str(
                round(process_step, 2)) + "%: ",
            "=" * (int(process_step) // 2), end="\033[0m")
    else:
        print("Epoch:%d Avg loss:%.4f          " % (epoch, avg_loss) + "Val progress:  " + pad + str(
            round(process_step, 2)) + "%: ", "=" * (int(process_step) // 2), end="")
    sys.stdout.flush()
