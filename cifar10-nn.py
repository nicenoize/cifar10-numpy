import numpy as np
import helper


class TwoLayerNeuralNetwork:
    """
    Ein 2-Layer 'fully-connected' neural network, d.h. alle Neuronen sind mit allen anderen
    verbunden. Die Anzahl der Eingabevektoren ist N mit einer Dimension D, einem 'Hidden'-Layer mit
    H Neuronen. Es soll eine Klassifikation über 10 Klassen (C) durchgeführt werden.
    Wir trainieren das Netzwerk mit einer 'Softmax'-Loss Funktion und einer L2 Regularisierung
    auf den Gewichtungsmatrizen (W1 und W2). Das Netzwerk nutzt ReLU Aktivierungsfunktionen nach
    dem ersten Layer.
    Die Architektur des Netzwerkes läßt sich abstrakt so darstellen:
    Eingabe - 'fully connected'-Layer - ReLU - 'fully connected'-Layer - Softmax

    Die Ausgabe aus dem 2.Layer sind die 'Scores' (Wahrscheinlichkeiten) für jede Klasse.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Intitialisierung des Netzes - Die Gewichtungsmatrizen und die Bias-Vektoren werden mit
        Zufallswerten initialisiert.
        W1: 1.Layer Gewichte (D, H)
        b1: 1.Layer Bias (H,)
        W2: 2.Layer Gewichte (H, C)
        b2: 2.Layer Bias (C,)

        :param input_size: Die CIFAR-10 Bilder haben die Dimension D (32*32*3).
        :param hidden_size: Anzahl der Neuronen im Hidden-Layer H.
        :param output_size: Anzahl der Klassen C.
        :param std: Skalierungsfaktoren für die Initialisierung (muss klein sein)
        :return:
        """
        self.W1 = std * np.random.randn(input_size, hidden_size)
        self.b1 = std * np.random.randn(hidden_size)
        self.W2 = std * np.random.randn(hidden_size, output_size)
        self.b2 = std * np.random.randn(output_size)

        # Ausgabe der Layer
        self.out_l1 = None
        self.out_l2 = None

    def softmax(self, z):
        z -= np.max(z)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def relu(self, w):
        return np.maximum(0.0, w)

    def relu_deriv(self, d, out):
        d[out <= 0] = 0
        return d

    def score(self, X):
        """
        Berechnet die Klassenscores. Schreibt den jeweiligen Output des Layers1 in self.out_l1 und
        den Output des Layers2 in self.out_l2
        :param X: Eingabebilder - Dimension (N - Anzahl der Bilder, D - Dimension eines Bildvektors)
        :return:
        """

        # TODO: Berechnen Sie den Forward-Schritt und geben Sie den Vektor mit Scores zurueck
        # Nutzen Sie die ReLU Aktivierungsfunktion im ersten Layer
        self.out_l1 = self.relu((np.dot(X, self.W1) + self.b1))
        # Berechnen Sie die Klassenwahrscheinlichkeiten unter Nutzung der softmax Funktion
        self.out_l2 = self.softmax((np.dot(self.out_l1, self.W2) + self.b2))

    def loss(self, batch_size, y, reg=0.0):
        """
        Berechnet den loss (Fehler) des 2-Layer-Netzes

        :param batch_size: Anzahl der Eingabebilder in einem Batch über den der Fehler normalisiert werden muss
        :param y: Vektor mit den Trainingslabels y[i] enthält ein Label aus X[i] und jedes y[i] ist ein
                  Integer zwischen 0 <= y[i] < C (Anzahl der Klassen)
        :param reg: Regulasierungsstärke
        :return: loss - normalisierter Fehler des Batches
        """
        loss = None
        # TODO: erganzen Sie den fehlen code
        # Berechnen Sie den Einzelfehler - pseudocode: -log( probabilities[y] )
        # Siehe Folie 8 in VSC-10 NN.pdf
        correct_logprobs = -np.log(self.out_l2[range(batch_size), y])
        # Compute the average loss
        loss = np.sum(correct_logprobs)/batch_size

        # Fehlerfunktion - Folie 9 in VSC-10 NN.pdf - Fehler muss noch uber alle Bilder normalisiert werden
        # loss = normalisierter data_loss (sum Li) / N + reg_loss ( lamda * 0.5*sum W1*W1 + lamda * 0.5*sum W1*W1)
        reg_loss = 0.5 * reg * np.sum(self.W1 * self.W1) + 0.5 * reg * np.sum(self.W2 * self.W2)
        loss += reg_loss

        return loss


    def forward(self, X, y, reg=0.0):
        """
        Führt den gesamten Forward Prozess durch. Dabei wird zunächst die Funktion score ausgeführt.
        Diese berechnet die Scores und schreibt das Ergebnis in die member variablen self.out_l1 und self.out_l2.
        Danach wird der Fehler in der Funktion loss berechnet und zurückgegeben
        :param X: Trainings bzw. Testset
        :param y: Labels des Trainings- bzw. Testsets
        :param reg: Regularisierungsstärke
        :return: loss
        """

        # Berechen Sie den score
        N, D = X.shape

        self.score(X)
        loss = self.loss(N, y, reg)

        return loss

    def backward(self, X, y, reg=0.0):
        """
        Backward pass- dabei wird der Gradient der Gewichte W1, W2 und der Biases b1, b2 aus den Ausgaben des Netzes
        berechnet und die Gradienten der einzelnen Layer als ein Dictionary zurückgegeben.
        Zum Beispiel sollte grads['W1'] die Gradienten von self.W1 enthalten (das ist eine Matrix der gleichen Größe
        wie self.W1.
        :param X:
        :param y:
        :param reg:
        :return:
        """

        # TODO
        # Backward pass: Berechnen Sie die Gradienten
        N, D = X.shape

        # Füllen Sie das Dictionary grads['W2'], grads['b2'], grads['W1'], grads['b1']
        grads = {}

        # Berechnen Sie zuerst den Gradienten der Scores
        dscores = self.out_l2
        dscores[range(N), y] -= 1
        dscores /= N
        # dann den Gradienten W2 und b2
        grads['W2'] = (self.out_l1.T).dot(self.out_l2)
        grads['b2'] = np.sum(dscores, axis=0, keepdims=True)
        delta2 = self.out_l2.dot(self.W2.T)
        # diesen müssen Sie zurückführen in den Hidden-Layer
        # Zurückführung der ReLU Nicht-Linearitaet
        delta2 = self.relu_deriv(delta2, self.out_l1)
        # Final dann den Gradienten auf W1 und b1
        grads['W1'] = np.dot(X.T, delta2)
        grads['b1'] = np.sum(delta2, axis=0)
        # diesen müssen Sie zurückführen in den Hidden-Layer

        # Zurückführung der ReLU Nicht-Linearitaet


        # Final dann den Gradienten auf W1 und b1
        #grads['W1']
        #grads['b1']

        # Dann wird noch die Regularisierung dazugepackt
        grads['W2'] += reg * self.W2
        grads['W1'] += reg * self.W1

        return grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=1000, verbose=False):
        """
        Training des Neuronalen Netzwerkes unter der Nutzung des iterativen
        Optimierungsverfahrens Stochastic Gradient Descent
        Train this neural network using stochastic gradient descent.

        :param X: Numpy Array der Größe (N,D)
        :param y: Numpy Array der Größe (N,) mit den jeweiligen Labels y[i] = c. Das bedeutet, dass X[i] das label c hat
                  mit 0 <= c < C
        :param X_val: Numpy Array der Größe (N_val,D) mit den Validierungs-/Testdaten
        :param y_val: Numpy Array der Größe (N_val,) mit den Labels für die Validierungs-/Testdaten
        :param learning_rate: Faktor der Lernrate für den Optimierungsprozess
        :param learning_rate_decay: gibt an, inwieweit die Lernrate in jeder Epoche angepasst werden soll
        :param reg: Stärke der Regularisierung
        :param num_iters: Anzahl der Iterationen der Optimierung
        :param batch_size: Anzahl der Trainingseingabebilder, die in jedem forward-Schritt mitgegeben werden sollen
        :param verbose: boolean, ob etwas ausgegeben werden soll
        :return: dict (fuer die Auswertung) - enthält Fehler und Genauigkeit der Klassifizierung für jede Iteration bzw. Epoche
        """
        num_train = X.shape[0]
        iterations_per_epoch = int(max(num_train / batch_size, 1))

        # Wir nutzen einen Stochastischen Gradient Decent (SGD) Optimierer um unsere
        # Parameter W1,W2,b1,b2 zu optimieren
        loss_history = []
        loss_val_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ############################
            # TODO: Erzeugen Sie einen zufälligen Batch der Größe batch_size
            # aus den Trainingsdaten und speichern diese in X_batch und y_batch
            # Create a random minibatch of training data and labels, storing  #
            # Gleiche Indizes
            idx = np.random.choice(np.arange(len(X_train)), batch_size, replace=False)
            #idx = np.arange(0, 200,)
            X_batch = X_train[idx, :]
            y_batch =  y[idx]#np.random.choice(y, batch_size, replace=False)

            ############################

            # Berechnung von loss und gradient für den aktuellen Batch
            loss = self.forward(X_batch, y_batch, reg=reg)
            grads = self.backward(X_batch, y_batch, reg=reg)

            # Merken des Fehlers für den Plot
            loss_history.append(loss)
            # Berechnung des Fehlers mit den aktuellen Parametern (W, b)
            # mit dem Testset
            loss_val = self.forward(X_val, y_val, reg=reg)
            loss_val_history.append(loss_val)

            ############################
            # TODO: Nutzen Sie die Gradienten aus der Backward-Funktion und passen
            # Sie die Parameter an (self.W1, self.b1 etc). Diese werden mit der Lernrate
            # gewichtet
            #print("LEARNING RATE * DICT", )
            self.W1 = self.W1 - (learning_rate * (grads['W1']))
            self.b1 = self.b1 - (learning_rate * (grads['b1']))
            self.W2 = self.W2 - learning_rate * (grads['W2'])
            self.b2 = self.b2 - learning_rate * (grads['b2'])


            ############################

            # Ausgabe des aktuellen Fehlers. Diese sollte am Anfang erstmal nach unten gehen
            # kann aber immer etwas schwanken.
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Wir überprüfen jede Epoche die Genauigkeit (von Trainingsset und Testset)
            # und dämpfen die Lernrate
            if it % iterations_per_epoch == 0:
                print('epoch done... ')
                # Überprüfung der Klassifikationsgenauigkeit
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Dämpfung der Lernrate
                learning_rate *= learning_rate_decay

        # Zum Plotten der Genauigkeiten geben wir die
        # gesammelten Daten zurück
        return {
            'loss_history': loss_history,
            'loss_val_history': loss_val_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Benutzen Sie die trainierten Gewichte des 2-Layer-Netzes um die Klassen für das
        Validierungsset vorherzusagen. Dafür müssen Sie für das/die Eingabebilder X nur
        die Scores berechnen. Der höchste Score ist die vorhergesagte Klasse. Dieser wird in y_pred
        geschrieben und zurückgegeben.

        :param X: Numpy Array der Größe (N,D)
        :return: y_pred Numpy Array der Größe (N,) die die jeweiligen Labels für alle Elemente in X enthaelt.
        y_pred[i] = c bedeutet, das fuer X[i] die Klasse c mit 0<=c<C vorhergesagt wurde
        """
        y_pred = None
        # Compute the forward pass
        scores = None
        self.out_l1 = self.relu((np.dot(X, self.W1) + self.b1))
        # Berechnen Sie die Klassenwahrscheinlichkeiten unter Nutzung der softmax Funktion
        self.out_l2 = self.softmax((np.dot(self.out_l1, self.W2) + self.b2))
        # Calculate second layer
        scores = self.out_l2

        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        summation = np.sum(exp_scores, axis=1, keepdims=True)
        probabilities = exp_scores / summation

        y_pred = np.argmax(probabilities, axis=1)


        ############################

        return y_pred



if __name__ == '__main__':

    # TODO: Laden der Bilder. Hinweis - wir nutzen nur die Trainigsbilder zum Trainieren und die
    # Validierungsbilder zum Testen.
    X_train, y_train, X_val, y_val = helper.prepare_CIFAR10_images()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)

    # Grösse der Bilder
    input_size = 32*32*3
    # Anzahl der Klassen
    num_classes = 10

    #############################################
    # Hyperparameter
    #############################################

    # TODO: mit diesen Parametern sollten Sie in etwa auf eine
    # Klassifikationsgenauigkeit von 43% kommen. Optimieren Sie die
    # Hyperparameter um die Genauigkeit zu erhöhen (bitte tun sie das
    # systematisch und nicht einfach durch probieren - also z.B. in einem
    # for-loop eine Reihe von Parametern testen und die Einzelbilder abspeichern)
    best_val = -1

    hidden_size = [50, 300]  # Anzahl der Neuronen im Hidden Layer
    num_iters = 3000 # Anzahl der Optimierungsiterationen
    batch_size = 400  # Eingabeanzahl der Bilder
    learning_rate = [1e-3, 1e-4]  # Lernrate
    learning_rate_decay = 0.95  # Lernratenabschwächung
    reg = [0.5, 1]  # Regularisierungsstärke


    for hs in hidden_size:
        for lr in learning_rate:
            for r in reg:
                    net = TwoLayerNeuralNetwork(input_size, hs, num_classes)
                    # Train the network
                    stats = net.train(X_train, y_train, X_val, y_val,
                                    num_iters=num_iters, batch_size=batch_size,
                                    learning_rate=lr, learning_rate_decay=learning_rate_decay,
                                    reg=r, verbose=True)
                    train_acc = (net.predict(X_train) == y_train).mean()
                    val_best = (net.predict(X_val) == y_val).mean()
                    print("Train acc: ", train_acc)
                    print("Val acc: ", val_best)
                    if val_best > best_val:
                        best_val = val_best
                        values = []
                        values.append(hs)
                        values.append(lr)
                        values.append(r)
                        print('Current best acc: ', best_val)
                        print("VALUES: ", values)
                    print('Final training loss: ', stats['loss_history'][-1])
                    print('Final validation loss: ', stats['loss_val_history'][-1])
                    print('Used values: hiddensize ', hs, 'learning rate: ', lr, 'reg: ', r)
                    print('Final validation accuracy: ', stats['val_acc_history'][-1])

    print('Best Accuracy: ', best_val)
    print('Best values: \nHidden_size: ', values[0],'\nLearning Rate: ', values[1],'\nReg: ', values[2])
    net = TwoLayerNeuralNetwork(input_size, 300, num_classes)
    # Generate best nets
    # 55.1% Accuracy mit hiddensize: 300 learning rate:  0.001 reg:  0.5
    stats = net.train(X_train, y_train, X_val, y_val, num_iters=num_iters, batch_size=batch_size, learning_rate=values[1], learning_rate_decay=learning_rate_decay, reg=values[2], verbose=True)
    final_acc = (net.predict(X_val) == y_val).mean()
    print('Final Accuracy reached: ', final_acc)
    helper.plot_net_weights(net)
    helper.plot_accuracy(stats)
    helper.plot_loss(stats)
