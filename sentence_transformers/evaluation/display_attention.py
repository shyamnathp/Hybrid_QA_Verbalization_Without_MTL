import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def display_attention(sentence, translation, attention, fileName, n_heads = 12, n_rows = 4, n_cols = 3):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(60,100))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+[t.lower() for t in sentence], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(('/data/premnadh/Hybrid-QASystem-New/Hybrid-QASystem/Utils/{}.png').format(fileName))
    plt.close()

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.savefig(('/data/premnadh/Hybrid-QASystem-New/Hybrid-QASystem/Utils/train_val_loss.png'))
    plt.close()
