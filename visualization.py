
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=1.2)


def print_model_weights(model):
    """Display network architecture and weights"""
    print(model.summary())
    print('\n\nmodel weights:')
    for layer in model.layers:
        print('kernel:')
        try:
            print(np.squeeze(layer.get_weights()[0]).transpose(2,0,1))
        except:
            print(np.squeeze(layer.get_weights()[0]))
        print('bias:', layer.get_weights()[1], end='\n\n')


def predict_glider(life, model):
    """Make predictions on Glider for 25 steps"""
    fig, axs = plt.subplots(figsize=(16,6), nrows=2, ncols=6)
    life.setup_state(init='glider')
    pred_state = life.state
    axs[0,0].imshow(life.state, cmap='viridis', vmin=0., vmax=1.)
    axs[1,0].imshow(pred_state, cmap='viridis', vmin=0., vmax=1.)
    axs[0,0].set_ylabel('Game of Life')
    axs[1,0].set_ylabel('Model')
    axs[1,0].set_xlabel('step 0')
    for step in range(25):
        life.step()
        pred_state = model(np.expand_dims(pred_state, axis=(0,-1))).numpy().squeeze()
        if not (step + 1) % 5:
            axs[0,(step+1)//5].imshow(life.state, cmap='viridis', vmin=0., vmax=1.)
            axs[1,(step+1)//5].imshow(pred_state, cmap='viridis', vmin=0., vmax=1.)
            axs[1,(step+1)//5].set_xlabel(f'step {step+1}')
    plt.grid(None)
    plt.setp(axs, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()
