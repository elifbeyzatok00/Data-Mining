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
