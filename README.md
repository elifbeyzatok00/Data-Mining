# Veri Madenciliği (Data Mining)

Veri madenciliği, büyük miktarda veri içerisindeki desenleri, bilgiyi ve anlamlı ilişkileri keşfetmek için kullanılan disiplinlerarası bir alanıdır. Genellikle istatistiksel analiz, makine öğrenimi, yapay zeka ve veritabanı yönetimi gibi alanlardan faydalanır. Veri madenciliği, veri içerisindeki gizli bilgileri ortaya çıkarmak, tahminler yapmak ve karar destek sistemlerini güçlendirmek için kullanılır.

🔗[Yapay Zekaya Giris Konu Anlatımı](https://github.com/elifbeyzatok00/Data-Mining/blob/main/Yapay_Zekaya_Giris.ipynb)

Veri madenciliğinin ana hedefleri şunlardır:

1. **Desen Keşfi:** Büyük veri kümelerindeki tekrarlanan desenleri ve ilişkileri tespit etmek.

2. **Tahmin:** Gelecekteki olayları veya değerleri tahmin etmek için verilerden modeller oluşturmak.

3. **Gruplama:** Benzer özelliklere sahip veri noktalarını gruplayarak segmentasyon yapmak.

4. **Yapısal Analiz:** Veri kümelerindeki yapının ve ilişkilerin anlaşılmasını sağlamak.

5. **Karar Destek:** Karar verme süreçlerini optimize etmek ve bilgiye dayalı kararlar almak için veri analizi ve modelleme kullanmak.

Veri madenciliği genellikle aşağıdaki adımları içeren bir süreçtir:

1. **Veri Toplama:** İlgili veri kaynaklarından veri toplanması.

2. **Veri Hazırlığı:** Veri temizleme, önişleme ve uygun formata getirme adımlarıyla veri hazırlığının yapılması.

3. **Modelleme:** Makine öğrenimi, istatistiksel analiz ve diğer teknikler kullanılarak veriye model uydurma.

4. **Değerlendirme:** Oluşturulan modellerin performansının değerlendirilmesi ve gerektiğinde iyileştirilmesi.

5. **Dağıtım:** Sonuçların yorumlanması ve gerektiğinde uygulamaya geçirilmesi.

Veri madenciliği, birçok endüstride kullanılan bir araçtır. Pazarlama, finans, sağlık, perakende, telekomünikasyon ve diğer birçok sektörde müşteri davranışlarını anlama, risk analizi yapma, hedefleme ve segmentasyon gibi birçok farklı alanda kullanılır.


### Makine Öğrenimi Algoritmaları
![image](https://github.com/elifbeyzatok00/Data-Mining/assets/102792446/5ac8d548-7bb1-4265-9f73-c71dddcd268d)

Makine öğrenimi algoritmaları, veriye dayalı örüntüleri tanımlamak, tahminler yapmak veya kararlar vermek için kullanılan matematiksel modellerdir. Bu algoritmalar, genellikle öğrenme sürecinde veriye adapte olarak performanslarını artırırlar. İşte yaygın olarak kullanılan bazı makine öğrenimi algoritmaları ve detayları:

1. **Denetimli Öğrenme Algoritmaları:**
   - **Doğrusal Regresyon:** Bağımlı ve bağımsız değişkenler arasındaki ilişkiyi modellemek için kullanılır. Örneğin, gelir ve harcama arasındaki ilişkiyi inceleyebilirsiniz.
   - **Lojistik Regresyon:** Sınıflandırma problemleri için kullanılır. Örneğin, bir e-postanın spam olup olmadığını belirlemek için kullanılabilir.
   - **Karar Ağaçları:** Karar ağaçları veri kümesini sınıflandırmak veya regresyon yapmak için ağaç benzeri bir yapı kullanır. Basit ve yorumlanabilir modeller sağlarlar.
   - **Destek Vektör Makineleri (SVM):** Bir veri kümesini sınıflandırmak veya regresyon yapmak için kullanılan bir algoritmadır. Belirli bir hiperdüzlemi oluşturarak sınıflar arasındaki en geniş boşluğu (margin) maksimize etmeye çalışır.
   - **K-En Yakın Komşu (KNN):** Bir veri noktasını etiketlemek için yakındaki veri noktalarının çoğunluğuna dayanır. Bu algoritma basit ve kullanımı kolaydır.

   🔗[Denetimli Öğrenme Konu Anlatımı](https://github.com/elifbeyzatok00/Data-Mining/blob/main/Denetimli_Ogrenme.pptx)

2. **Denetimsiz Öğrenme Algoritmaları:**
   - **K-Means Kümeleme:** Veri noktalarını belirli sayıda kümeye (clusters) bölmek için kullanılır. Her kümenin merkezi, o kümenin veri noktalarının ortalamasıdır.
   - **Hiyerarşik Kümeleme:** Bir veri kümesini hiyerarşik olarak kümeler halinde böler. Bu kümeler, bir ağaç yapısı oluşturur ve farklı düzeylerdeki kümeleme seviyelerini gösterir.
   - **Gauss Karışım Modelleme (GMM):** Her biri bir normal dağılıma sahip olan birden fazla bileşenin bir karışımı olarak veriyi modellemek için kullanılır. Genellikle veriye gizli bir yapı uygular.
   - **Boyut Azaltma Algoritmaları:** Boyut azaltma, veri setinin boyutunu azaltarak veriyi daha anlaşılabilir ve işlenebilir hale getirir. Örneğin, PCA (Principal Component Analysis) ve t-SNE (t-distributed Stochastic Neighbor Embedding) gibi algoritmalar kullanılabilir.

3. **Pekiştirmeli Öğrenme:**
   - **Q-Learning:** Pekiştirmeli öğrenme için temel bir algoritmadır. Bir ajanın çevresiyle etkileşimde bulunarak bir görevi en iyi şekilde yerine getirmesini öğretir. Belli bir durumda alınabilecek aksiyonların değerini öğrenir.
   - **Policy Gradient:** Bu algoritma, doğrudan bir politika fonksiyonunu optimize ederek ajanın davranışını iyileştirmeye çalışır. Genellikle derin öğrenmeyle birlikte kullanılır.

4. **Semi-Supervised Learning (Yarı-Denetimli Öğrenme):**
   - Hem etiketli hem de etiketlenmemiş verilerin kullanıldığı bir öğrenme türüdür. Bu yaklaşım, sınıflandırma veya regresyon problemlerinde etiketli verinin sınıfını belirlemek için kullanılırken, aynı zamanda etiketlenmemiş veriden de yararlanır.

Bu, makine öğrenimi alanında sıkça kullanılan temel algoritmaların bazılarıdır. Her bir algoritmanın farklı avantajları, dezavantajları ve kullanım alanları vardır. Veri setinizin boyutu, doğası ve probleminizin gereksinimleri, hangi algoritmanın kullanılacağını belirlemede önemli faktörlerdir.

![image](https://github.com/elifbeyzatok00/Data-Mining/assets/102792446/e5a5f30d-cb7c-458c-9dd7-6977cd1ed21c)

![image](https://github.com/elifbeyzatok00/Data-Mining/assets/102792446/3e6e01e3-cd26-46f6-9c2b-f9dc3faa7804)

![image](https://github.com/elifbeyzatok00/Data-Mining/assets/102792446/4fdbdc4d-56a2-468d-be7d-d6fb5f9cbcc9)

🔗[Makine Ogrenmesi Modelleri Konu Anlatımı](https://github.com/elifbeyzatok00/Data-Mining/blob/main/Makine_Ogrenmesi_Modelleri.ipynb)



### Derin Öğrenme Algoritmaları

![image](https://github.com/elifbeyzatok00/Data-Mining/assets/102792446/19d08b58-61fa-4eee-a435-0ab94502544c)

Derin öğrenme, yapay sinir ağlarının çok katmanlı ve karmaşık yapılarını kullanarak öğrenme sürecini gerçekleştiren bir makine öğrenimi alt dalıdır. İşte derin öğrenme algoritmalarının başlıca tipleri ve detayları:

1. **Yapay Sinir Ağları (Artificial Neural Networks - ANN):**
   - Yapay sinir ağları, biyolojik sinir ağlarını temel alan bir modeldir. Çok katmanlı algılayıcılar, gizli katmanlar ve çıktı katmanlarından oluşur.
   - Her katman, giriş verisini işleyerek daha karmaşık özellikler öğrenir.
   - Geriye yayılım algoritması (Backpropagation), ağın hatalarını azaltmak için kullanılır.

   🔗[Yapay Sinir Aglari Giris Konu Anlatımı](https://github.com/elifbeyzatok00/Data-Mining/blob/main/Yapay_Sinir_Aglari_Giris.ipynb)

2. **Evrişimli Sinir Ağları (Convolutional Neural Networks - CNN):**
   - Görüntü ve video gibi yapısal veriler üzerinde başarıyla çalışan bir derin öğrenme modelidir.
   - Farklı özellik haritaları oluşturmak için evrişim (convolution) ve havuzlama (pooling) katmanlarından oluşur.
   - Görüntü sınıflandırma, nesne tespiti, yüz tanıma gibi görevlerde etkilidir.

3. **Rekurrent Sinir Ağları (Recurrent Neural Networks - RNN):**
   - Zamansal bağımlılıkları modellemek için tasarlanmıştır. Önceki zaman adımlarından gelen bilgileri hafızada saklarlar.
   - Özellikle zaman serisi verileri, metin verileri ve doğal dil işleme problemleri için kullanılır.
   - Geleneksel RNN yapılarının yanı sıra, uzun süreli bağımlılıkları ele almak için LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit) gibi modeller geliştirilmiştir.

4. **Özerk Kodlayıcılar (Autoencoders):**
   - Giriş verisini sıkıştırılmış bir temsilasyona dönüştüren ve ardından orijinal veriyi yeniden oluşturan sinir ağlarıdır.
   - Unsupervised learning'de kullanılırlar ve veri setindeki gizli yapıları öğrenmek için kullanılırlar.
   - Gürültülü girişlere dayanıklı olabilirler ve boyut azaltma gibi görevlerde kullanılabilirler.

5. **Derin Üretim Modelleri (Generative Deep Learning Models):**
   - Veri setinden yeni örnekler üreten modellerdir.
   - GANs (Generative Adversarial Networks), VAEs (Variational Autoencoders) ve RBMs (Restricted Boltzmann Machines) gibi modellerle temsil edilirler.
   - Örneğin, GAN'lar gerçekçi görüntüler üretmek için kullanılabilirler.

Bu algoritmalar, derin öğrenmenin temel yapı taşlarını oluşturur. Her biri farklı veri türlerini ve problemlerini ele almak için tasarlanmıştır. Gelişen teknoloji ve araştırmalarla, bu algoritmaların yeni varyasyonları ve iyileştirmeleri sürekli olarak geliştirilmektedir.


### Veri madenciliğinde kullanılan melez algoritmalar nelerdir? Birkaç algoritma birlikte nasıl kullanılır? Ne amaçla kullanılır?

Veri madenciliğinde kullanılan melez algoritmalar, genellikle birbirini tamamlayıcı özelliklere sahip olan farklı algoritmaların bir araya getirilmesiyle oluşturulur. Bu, genellikle daha iyi sonuçlar elde etmek, daha geniş bir veri yelpazesini kapsamak veya belirli zorlukları aşmak için yapılır. İşte bazı yaygın melez algoritmalar ve nasıl kullanıldıklarıyla ilgili örnekler:

1. **CNN+KNN (Convolutional Neural Network + K-Nearest Neighbors):**
   - **Kullanım Amacı:** Görüntü sınıflandırma gibi görevlerde yaygın olarak kullanılır. CNN, derin öğrenme modeli olarak özellik çıkarımı yapar ve görsel veriyi temsil ederken, KNN ise bu özellik vektörlerini kullanarak sınıflandırma işlemi yapar.
   - **Nasıl Kullanılır:** İlk adımda, bir CNN modeli eğitilir ve görüntülerden özellikler çıkarılır. Daha sonra, bu özellik vektörleri KNN algoritmasıyla birlikte kullanılarak sınıflandırma yapılır. Yani, CNN modeli görüntüleri özellik vektörlerine dönüştürür ve KNN, bu vektörler üzerinde sınıflandırma yapar.

2. **Random Forest + Gradient Boosting:**
   - **Kullanım Amacı:** Karar ağaçlarına dayalı yöntemlerin kombinasyonu, tahmin ve sınıflandırma problemlerinde genellikle daha yüksek doğruluk sağlamak için kullanılır.
   - **Nasıl Kullanılır:** İlk olarak, Random Forest algoritması kullanılarak birçok karar ağacı eğitilir. Daha sonra, bu ağaçların birleşiminden bir tahmin yapılır. Bu tahminler, Gradient Boosting algoritması kullanılarak daha fazla rafine edilir. Gradient Boosting, önceki tahminlerin hatalarını düzelterek modeli güçlendirir.

3. **K-Means + DBSCAN:**
   - **Kullanım Amacı:** Kümeleme problemlerinde kullanılır. K-Means, kümeleme işlemi için merkezi yöntemlerden biridir, ancak veri yapısının gürültülü veya karmaşık olduğu durumlarda yetersiz kalabilir. Bu durumda, DBSCAN gibi yoğunluğa dayalı bir kümeleme algoritması kullanılabilir.
   - **Nasıl Kullanılır:** İlk olarak, K-Means algoritması kullanılarak kümeleme yapılır. Daha sonra, DBSCAN gibi bir yoğunluk tabanlı algoritma kullanılarak kümeleme sonuçları daha fazla rafine edilebilir. Özellikle, K-Means'in oluşturduğu küme merkezlerini başlangıç noktası olarak kullanarak DBSCAN, daha kesin ve esnek kümeleme sonuçları üretebilir.

Bu melez algoritmalar, belirli veri madenciliği problemlerine daha iyi çözümler bulmak için kullanılır. Bunlar, tek başına kullanılan algoritmaların zayıf noktalarını dengelemek ve farklı özelliklerin avantajlarını bir araya getirmek için tasarlanmıştır.

Örnek:

İşte bir Python kod örneği, görüntü sınıflandırma için CNN (Convolutional Neural Network) ile KNN (K-Nearest Neighbors) algoritmalarının nasıl bir araya getirilebileceğini gösteriyor. Bu örnekte, MNIST el yazısı rakam veri kümesini kullanacağız. İlk olarak, CNN kullanarak el yazısı rakamları tanımak için bir model eğiteceğiz ve ardından bu modelin özellik vektörlerini çıkaracağız. Daha sonra, KNN algoritması kullanılarak bu özellik vektörlerini sınıflandıracağız.

```python
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier

# MNIST veri kümesini yükleme
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Veri kümesini yeniden şekillendirme ve normalleştirme
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# Etiketleri ikili sınıf matrislerine dönüştürme
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN modelini oluşturma
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# CNN modelinin çıktılarını kullanarak özellik vektörlerini çıkarın
extractor = Sequential(model.layers[:-1])

X_train_features = extractor.predict(X_train)
X_test_features = extractor.predict(X_test)

# KNN modelini oluşturma ve eğitme
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_features, np.argmax(y_train, axis=1))

# Test verilerini kullanarak tahmin yapma
predictions = knn_model.predict(X_test_features)

# Sonuçları değerlendirme
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print("Test doğruluğu:", accuracy)
```

Bu kod örneğinde, öncelikle MNIST veri kümesi yüklenir ve CNN modeli tanımlanır ve eğitilir. Daha sonra, bu modelin çıktılarını kullanarak özellik vektörleri çıkarılır ve KNN algoritması kullanılarak bu özellik vektörleri sınıflandırılır. Son olarak, sınıflandırma sonuçları değerlendirilir ve test doğruluğu hesaplanır.

### Veri madenciliği: Öznitelik seçim alg ile kodlanacak tasarım adımlarını py ile kodlamaya dönüştür. (Kod yok PSO entegrasyonu gerçekleştir)

Tabii, işte yorum satırları ile birlikte tüm kodu tek bir hücrede:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Veri setini oluştur
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Veri setini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parçacık Sürü Optimizasyonu (PSO) sınıfını tanımla
class Particle:
    def __init__(self, num_features):
        # Parçacık pozisyonunu rastgele seç
        self.position = np.random.choice([0, 1], size=num_features)
        # Parçacık hızını rastgele ata
        self.velocity = np.random.uniform(-1, 1, size=num_features)
        # Parçacıkın en iyi pozisyonunu ve skorunu tut
        self.best_position = self.position.copy()
        self.best_score = float('-inf')

class PSO:
    def __init__(self, num_particles, num_iterations):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.particles = []

    def optimize(self, X_train, X_test, y_train, y_test):
        num_features = X_train.shape[1]
        # Parçacıkları oluştur ve listeye ekle
        for _ in range(self.num_particles):
            particle = Particle(num_features)
            self.particles.append(particle)
        
        global_best_position = np.zeros(num_features)
        global_best_score = float('-inf')
        
        # PSO iterasyonlarını başlat
        for _ in range(self.num_iterations):
            # Her bir parçacık için optimize et
            for particle in self.particles:
                # Seçilmiş özellikleri al
                selected_features = np.where(particle.position == 1)[0]
                if len(selected_features) == 0:
                    continue
                
                # Sınıflandırıcıyı oluştur ve eğit
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train[:, selected_features], y_train)
                # Test seti üzerinde tahmin yap
                y_pred = clf.predict(X_test[:, selected_features])
                score = accuracy_score(y_test, y_pred)
                
                # En iyi pozisyonu ve skoru güncelle
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle.position.copy()
            
            # Hız ve pozisyon güncellemelerini hesapla
            for particle in self.particles:
                particle.velocity += 2 * np.random.random() * (particle.best_position - particle.position) + \
                                     2 * np.random.random() * (global_best_position - particle.position)
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, 0, 1)
        
        return global_best_position

# PSO'yu kullanarak öznitelik seçimini gerçekleştir ve sonuçları yazdır
num_particles = 20
num_iterations = 50

pso = PSO(num_particles, num_iterations)
selected_features = pso.optimize(X_train, X_test, y_train, y_test)
print("Selected Features:", np.where(selected_features == 1)[0])
```

Bu kod parçacıklar arası optimizasyon kullanarak öznitelik seçimi gerçekleştirir ve seçilen özniteliklerin indekslerini yazdırır.

### Veri madenciliği: Öznitelik seçim alg ile DDOS saldırı tespiti ML modeli geliştir, veri setini Kaggle’dan çek. (Kod yok PSO entegrasyonu gerçekleştir)

🔗[DDOS Attack Detection ML Model]()
