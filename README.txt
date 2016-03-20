---------------------------
Convolutional Neural Network written in Python 3 with Theano.
Testing is done on the MNIST dataset of 70000 handwritten digits.
Depends on Python 3, SciPy/NumPy, Theano and CUDA. Please make sure you
have those installed and working!

Run runner.bash or the following command:
THEANO_FLAGS='device=gpu' python3 thCNN.py

After the run, please send me the generated logCNN.txt to
csxeba@gmail.com

In case you get an error message like such,
ERROR (theano.sandbox.cuda): nvcc compiler not found on $PATH. Check your nvcc installation and try again.
then you probably don't have a working CUDA installation.
Please inform me if this happens!

Thank you for the help!

---------------------------
Konvolúciós neurális hálózat Python 3 nyelven, Theano-val.
A tesztelés a 70000 kézzel írott számjegyet tartalmazó MNIST adatsoron
történik.
Programfüggőségek: Python 3, SciPy/NumPy, Theano és CUDA. Légy szíves
győződj meg róla, hogy ezek telepítve vannak nálad és működnek.

Futtasd a runner.bash fájlt vagy a következő parancsot:
THEANO_FLAGS='device=gpu' python3 thCNN.py

Futás után kérlek küldd el nekem a generált logCNN.txt fájlt ide:
csxeba@gmail.com

Amennyiben a következő hibaüzenet megjelenne,
ERROR (theano.sandbox.cuda): nvcc compiler not found on $PATH. Check your nvcc installation and try again.
akkor feltehetőleg nincs működő CUDA telepítésed.
Kérlek jelezd nekem, ha ez történik!

Köszönöm a segítséget!
