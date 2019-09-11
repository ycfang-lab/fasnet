import matplotlib.pyplot as plt
import time


def plot_det_curve(far, frr, info=''):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_yscale('logit')
    ax.set_xscale('logit')
    ax.set_xlim([0.002, 0.99])
    ax.set_ylim([0.002, 0.99])
    ax.set_xlabel("False Rejection Rate (%)")
    ax.set_ylabel("False Accaptance Rate (%)")
    ax.set_title('DET Curve')
    ax.spines['left']._adjust_location()
    ax.spines['bottom']._adjust_location()
    ax.grid(True, linestyle='--')
    ax.minorticks_off()

    ticks = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.98, 0.99]
    plt.yticks(ticks, [100 * i if 100 * i < 1 else int(100 * i) for i in ticks])
    plt.xticks(ticks, [100 * i if 100 * i < 1 else int(100 * i) for i in ticks])

    ax.plot(frr, far, '--', lw=2)
    plt.show()

    # save image to eps
    loctime = time.localtime(time.time())
    strtime = time.strftime('%Y-%m-%d-%H-%M-%S', loctime)
    fig.savefig('-'.join([info, strtime]) + '.eps', format='eps', dpi=100)
