
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "VIJAY.S.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_4WV3YGFBKCO"
      },
      "source": [
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oP4tZQ9KpXAN"
      },
      "source": [
        "from keras.datasets import mnist\n",
        "data = mnist.load_data()"
      ],
      "execution_count": 81,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V-9HJflrqCCp"
      },
      "source": [
        "(x_train, y_train), (x_test, y_test) = mnist.load_data()"
      ],
      "execution_count": 83,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IieFX-bgqw1T",
        "outputId": "f93bfbf4-38dd-4efb-fdd1-dfa490556780"
      },
      "source": [
        "x_train[0].shape"
      ],
      "execution_count": 84,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(28, 28)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 84
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8zLrClcLq794"
      },
      "source": [
        "x_train = x_train.reshape((x_train.shape[0], 28*28)).astype('float32')\n",
        "x_test = x_test.reshape((x_test.shape[0], 28*28)).astype('float32')"
      ],
      "execution_count": 85,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Q1QmhztFtcyB"
      },
      "source": [
        "x_train = x_train/255\n",
        "x_test = x_test/255"
      ],
      "execution_count": 86,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6t_snTIQuMH2",
        "outputId": "0402f5f4-da79-4eef-ec61-b5953086287d"
      },
      "source": [
        "from keras.utils import np_utils\n",
        "\n",
        "print(y_test.shape)\n",
        "\n",
        "y_train = np_utils.to_categorical(y_train)\n",
        "y_test = np_utils.to_categorical(y_test)\n",
        "\n",
        "nun_classes = y_test.shape[1]\n",
        "print(y_test.shape)"
      ],
      "execution_count": 87,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(10000,)\n",
            "(10000, 10)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VqNeDVKgxvrN"
      },
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Dense"
      ],
      "execution_count": 88,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "v9WUk4DfyaWv"
      },
      "source": [
        "model = Sequential()\n",
        "model.add(Dense(32, input_dim = 28*28, activation = 'relu'))\n",
        "model.add(Dense(64, activation ='relu'))\n",
        "model.add(Dense(10, activation = 'softmax'))"
      ],
      "execution_count": 89,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aVWiAkJT0tOL"
      },
      "source": [
        "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])"
      ],
      "execution_count": 90,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SKVLtsak1SxS",
        "outputId": "36a82a7a-6486-4bb8-e29a-996bdd64b70d"
      },
      "source": [
        "model.summary()"
      ],
      "execution_count": 92,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"sequential_9\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "dense_18 (Dense)             (None, 32)                25120     \n",
            "_________________________________________________________________\n",
            "dense_19 (Dense)             (None, 64)                2112      \n",
            "_________________________________________________________________\n",
            "dense_20 (Dense)             (None, 10)                650       \n",
            "=================================================================\n",
            "Total params: 27,882\n",
            "Trainable params: 27,882\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sRfRZpG41f8z",
        "outputId": "919e443c-8c20-493f-d087-1ad61e99a063"
      },
      "source": [
        "model.fit(x_train, y_train, epochs=10, batch_size=100)"
      ],
      "execution_count": 93,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/10\n",
            "600/600 [==============================] - 2s 2ms/step - loss: 0.7853 - accuracy: 0.7858\n",
            "Epoch 2/10\n",
            "600/600 [==============================] - 1s 2ms/step - loss: 0.2133 - accuracy: 0.9391\n",
            "Epoch 3/10\n",
            "600/600 [==============================] - 1s 2ms/step - loss: 0.1589 - accuracy: 0.9528\n",
            "Epoch 4/10\n",
            "600/600 [==============================] - 1s 2ms/step - loss: 0.1324 - accuracy: 0.9606\n",
            "Epoch 5/10\n",
            "600/600 [==============================] - 1s 2ms/step - loss: 0.1129 - accuracy: 0.9667\n",
            "Epoch 6/10\n",
            "600/600 [==============================] - 1s 2ms/step - loss: 0.1006 - accuracy: 0.9695\n",
            "Epoch 7/10\n",
            "600/600 [==============================] - 1s 2ms/step - loss: 0.0882 - accuracy: 0.9722\n",
            "Epoch 8/10\n",
            "600/600 [==============================] - 1s 2ms/step - loss: 0.0775 - accuracy: 0.9766\n",
            "Epoch 9/10\n",
            "600/600 [==============================] - 1s 2ms/step - loss: 0.0690 - accuracy: 0.9790\n",
            "Epoch 10/10\n",
            "600/600 [==============================] - 1s 2ms/step - loss: 0.0670 - accuracy: 0.9799\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tensorflow.python.keras.callbacks.History at 0x7f9aa9c22190>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 93
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZaHJOhzi22kH",
        "outputId": "b4b7d179-75d0-4948-8de4-21fd04a4dffb"
      },
      "source": [
        "scores = model.evaluate(x_test, y_test)\n",
        "print(scores)"
      ],
      "execution_count": 94,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "313/313 [==============================] - 0s 925us/step - loss: 0.1085 - accuracy: 0.9679\n",
            "[0.10846977680921555, 0.9678999781608582]\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}
