import tensorflow as tf
import numpy as np
#Tensorflow udostępnia nowe typy zmiennych.
#Tf.Tensor reprezentuje wielowymiarową tablicę elementów.


#Utwórzmy tensor z wartościami typu float
tens = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print(tens.dtype)
#A teraz utwórzmy tensor z wartościami int
tens_int = tf.constant([[1, 2], [3, 4]])
print(tens_int.dtype)

#Jak widać, konstruktor automatycznie wybiera typ danych.
#Jeśli chcemy mieć pewność, że dany typ będzie miał zastosowanie, musimy zdefiniować (podobnie jak w NumPy) argument dtype.
tens_float32 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
print(tens_float32.dtype)

#Warto wspomnieć, że typem tej zmiennej jest EagerTensor
type(tens) 

#Zmienna przechowuje nie tylko wartości, ale także niektóre metadane
print(tens)

#Jeśli chcemy odwołać się do czystych wartości, możemy użyć metody numpy.
print(tens.numpy())
type(tens.numpy())


#W podanym przykładzie użyliśmy funkcji tf.constant do utworzenia tensora. W rzeczywistości pakiet tensorflow daje nam więcej opcji.
#tf.Variable vs tf.constant
tens = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
#tf.Variable jest podobne do tf.constant. Główną różnicą jest to, że tf.constant jest niezmienne (jego wartości nie można zmienić. Jeśli chcesz zmienić jego wartości, musisz zastąpić cały obiekt).
#Warto również wspomnieć, że nie można zmieniać wartości, jak w obiektach typu ndarray.
#Próba wykonania tens[0][0] = 2 lub tens.numpy()[0][0] spowoduje błąd.
#Aby zastąpić każdą wartość, możemy użyć funkcji assign
tens.assign([[-1,-2], [5,6]])
#Jeśli chcemy dodać dwa tensory możemy użyć assign_add
tens.assign_add([[1,2], [5,6]])
#a jeśli chcemy je odjąć, możemy użyć assign_sub
tens.assign_sub([[1,2], [5,6]])

#Aby pomnożyć tensory używamy matmul
tf.matmul(tens, tens)

#Mnożenie element z elementem można wykonać za pomocą
tf.multiply(tens, tens)


#Aby zmienić tylko wybrane części oryginalnego tensora, możemy zastosować funkcję „where”. Ta funkcja przyjmuje trzy argumenty:
#condtion: obiekt podobny do tensora, który
#x: tensor, którego wartości są używane, jeśli parametr warunku ma wartość różną od zera (True)
#y: tensor, którego wartości są używane, jeśli parametr warunku ma wartość zerową (False)

#Jeśli chcemy zmienić wartość w drugiej kolumnie naszego tensora na 0, możemy wykonać następujące kroki:
#Utwórz tablicę True
condition = np.repeat(True, tens.numpy().size).reshape(tens.shape)
#Umieść wartość Flase w drugiej kolumnie
condition[:, 1] = False
#Funkcja „where” zwraca nowy Tensor z oczekiwanymi wartościami
tens2 = tf.where(condition, tens, 0)

#Zamiast tworzyć własną macierz warunków, możemy zastosować przydatną funkcję, która zwraca tensor o wartościach logicznych
    #tf.math.less
    #tf.math.greater
    #tf.math.less_equal
    #tf.math.greater_equal
    #tf.math.is_nan


#Funkcja 'where' może być również używana do uzyskiwania indeksów elementów niezerowych w tensorze.
#Porównaj
tf.where(tens2)
z
tf.where(tens)

#Aby wyciąć część tensora, możemy również użyć funkcji gather, która przyjmuje 3 parametry:
#tens: tensor do wycięcia
#indices: lista elementów, które mają utworzyć dziesiątki
#axis: oś, z której chcemy wyciąć
#Pierwsza kolumna
tf.gather(tens, [0], axis=1)
#Druga kolumna
tf.gather(tens, [1], axis=1)
#Pierwszy rząd
tf.gather(tens, [0], axis=0)
#Drugi rząd
tf.gather(tens, [1], axis=0)

#Różne typy tensorów

#tf.RaggedTensor to tensory, których wycinki mogą mieć różną długość.
rt = tf.RaggedTensor.from_row_splits(values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])

#tf.sparse.SparseTensor są używane dla tensora z wieloma zerami
#Tle poniższe polecenie tworzy tensor złożony z zer, z wyjątkiem indeksów [0,0] i [5,0]. Wartości na tych pozycjach są podane przez parametr 'values'. Parametry 'dense_shape' definiują kształt tensora.
st1 = tf.SparseTensor(indices=[[0, 0], [5, 0]],
                      values=[1, 2],
                      dense_shape=[10, 3])

#Jeśli spróbujemy wydrukować jego wartość, otrzymamy tylko wartości różne od zera.
print(st1.values)

#Teraz utwórzmy drugi tensor rzadki, z wartością różną od zera (jedną wspólną) pozycją [5,0] i (różną) pozycją [5,2].
st2 = tf.SparseTensor(indices=[[5, 0], [5, 2]],
                      values=[10, 20],
                      dense_shape=[10, 3])
print(st2.values)

#Aby dodać te rzadkie tensory, musimy użyć specjalnej funkcji
st_r = tf.sparse.add(st1, st2)

#Jak widać, są trzy wartości różne od zera.
print(st_r.values)

#Możemy przekonwertować go na EagerTensor za pomocą funkcji sparse.to_dense().
type(tf.sparse.to_dense(st_r))




tf.newaxis # można użyć do dodania nowej osi do tablicy/tensora

arr = np.arange(100).reshape((50,2)) #tablica 50 wierszy i 2 kolumn

arr_expand_tf = arr[..., tf.newaxis] 

tf.reshape(arr_expand_tf, [1,50,2])


arr_expand_tf.shape #3-osiowa tablica (tensor) z 50 wierszami, 2 kolumnami i 1 dodatkowym kanałem

#To samo można zrobić za pomocą expand_dims

tf.expand_dims(arr, axis=0).shape