# DDNet (CENG-514, 2022): 2026 Yayınlanabilirlik Değerlendirmesi ve Makale Planı

*Hazırlanma tarihi: 25 Eylül 2026. Bu klasördeki tüm sayılar repodaki veri ve kodla yeniden üretilebilir (bkz. Ek B). Mevcut kodlara dokunulmadı; yalnızca `analysis/` klasörü eklendi.*

---

## 0. Kısa cevap

**Proje neydi?** DDNet, ilaç–ilaç etkileşimi (DDI) tahminini ikili bir bağlantı tahmini problemi olarak ele alıyordu. Adımlar şunlardı:
- Morgan parmak izi kosinüs benzerliği ile KEGG SIMCOMP benzerliği birleştirilerek homojen bir **ilaç benzerlik ağı** kuruluyordu.
- Her ilacın 1-hop/2-hop ego alt-grafında **ayrı ayrı node2vec** çalıştırılıyor ve merkez düğümün vektörü ilaç özelliği olarak alınıyordu.
- Buna 1–3 uzunluklu yollardaki ağırlık çarpımlarının toplamı ("meta-path") ekleniyordu.
- Sonuçta RF / GB / FFNN ile DrugBank etiketleri öğreniliyordu.
- Rapordaki manşet sonuç şuydu: 289 ilaçlık MP kümesinde FFNN ile F1 0.85 / MCC 0.72, yani "HIN-DDI'yi geçtik".

**Mevcut haliyle yayınlanabilir mi? Hayır.** Aşağıdaki üç gerekçenin her biri tek başına yeterli ve hepsi bağımsız olarak doğrulandı:

1. **Skorlar yöntemi değil, veri seçimini ve protokolü ölçüyor.**
   - Seçilen ilaçlar DrugBank'in en çok etkileşen ilaçları: medyan derece 1.160–1.265, genel medyan ise 578.
   - Bu yüzden pozitif oranı %45–58 (tüm DrugBank'te %14).
   - Hiç özellik kullanmayan, yalnızca ilacın eğitimdeki etkileşim sıklığına bakan bir "derece" skoru AUROC 0.82–0.87 veriyor.
   - Aynı FFNN ile rastgele Gauss vektörleri node2vec'ten **daha iyi** sonuç veriyor (RF ile node2vec ~0.05 önde). İlaç kimliği (one-hot) ile eğitilen bir MLP 0.96–0.98 AUROC ile her şeyi geçiyor.
   - İlaçlar test kümesinde gerçekten yeni olduğunda (soğuk başlangıç) DDNet özellikleri düz Morgan parmak izinden ayırt edilemiyor. Yol özellikleri hiçbir şey eklemiyor.
2. **Yöntemin hiçbir bileşeni 2026'da özgün değil.**
   - Benzerlik ağı + gömme + sınıflandırıcı yaklaşımı daha önce yayımlanmış: HIN-DDI grubunun 2021 BigData makalesi ve HAN-DDI'deki temel çizgiler.
   - Yol-çarpımı skoru DASPfind / Katz indeksidir.
   - Yerel alt-graf öğrenmesi SEAL, SumGNN, KnowDDI ve CSSE-DDI'de var.
   - "DDNet" adı da dolu (IEEE TMI 2018 CT modeli ve başkaları).
3. **Raporlanan sayılar kodla tutarlı değil.**
   - D1/D2 tabloları, aritmetik olarak yalnızca test pozitif oranı 0.61–0.73 ile tutarlı. Bu oran IQR aykırı-değer filtresinden geçen 71/73 ilaçlık alt kümelere karşılık geliyor ve raporda belirtilmemiş.
   - Git geçmişindeki filtresiz çalıştırmalar MCC 0.28–0.41 veriyor; raporda 0.68–0.71 yazıyor.
   - HIN-DDI karşılaştırması geçersiz: farklı ilaç kümesi, %50 pozitif oranı, tek bölme ve sert etiketten hesaplanmış AUPR.
   - "Yeni ilaç" deneyi (D1→D2) şansa eşit. D1 ve D2 46 ilacı paylaşıyor ve embedding uzayları hizalı değil.

**Biraz eforla makaleye dönüşür mü? Evet, ama "DDNet yöntemi" olarak değil.**
- En güçlü ve en fizibıl seçenek bir **değerlendirme-bilimi makalesi**: "Soğuk-başlangıç DDI benchmark'ları gerçekte neyi ölçüyor? İlaç düzeyi eğilimler (perpetrator/victim) ve çifte özgü etkileşim."
- DDNet'in öz-denetimi bu makalede "Kutu 1" vaka analizi olarak yer alır.
- Tahmini maliyet ≈ 350 kişi-saat (iki yarı zamanlı yazar için ~18–20 hafta). Birincil hedef *Briefings in Bioinformatics*.
- Dürüst not: hakem ajanlarının üçü de özgünlüğe 5/10 verdi. Bu yüksek etkili değil, **sağlam ve yayımlanabilir** bir makale.
- Planın dayandığı iki kilit pilot bulgu bağımsız olarak yeniden üretildi (bkz. §8.1):
  - DDI-Ben TWOSIDES S1'de eğitimsiz "bilinen ilacın derecesi" skoru ROC-AUC 87.5 veriyor. Nedeni negatiflerin kuruluş biçimi; partner-eşleşmeli negatiflerle skor 50'ye iniyor. Birebir doğrulandı.
  - DrugBank-86 S1'de eğitimsiz eklemeli bir kural makro-F1 ≈54–57 veriyor; yayımlanmış en iyi değer ≈57 (ikincil kaynaktan).

**Acil bir konu (makaleden bağımsız): lisans.**
- Herkese açık, MIT lisanslı bu repo DrugBank DDI tablosunun tamamını (`data/drug-drug_interaction_Drugbank.csv`, 2,77 M satır) ve DrugBank kimliklerini/SMILES/hedefleri içeriyor.
- DrugBank'in akademik verisi CC BY-NC 4.0 ve yeni şartları yeniden dağıtım için lisans istiyor.
- Bunu kaldırmak veya repoyu gizlemek sizin kararınız; ben dokunmadım (§9).

---

## 1. Nasıl değerlendirildi?

Çalışma üç çok-ajanlı tur halinde yürütüldü (toplam ~27 ajan). Her bulgu, onu üretenden bağımsız bir doğrulayıcı tarafından çürütülmeye çalışıldı.

| Tur | İçerik |
|---|---|
| 1. Araştırma + doğrulama (14 ajan) | 5 literatür ekseni (SOTA, en yakın önceki çalışmalar/özgünlük, değerlendirme metodolojisi, 2024–26 öncü yaklaşımlar + CROssBAR, mecralar/lisans); statik kod denetimi; ampirik yeniden değerlendirme. Her birine ayrı bir doğrulayıcı (kaynak kontrolü, çürütme girişimi, sıfırdan yeniden uygulama) atandı. |
| 2. Makale tasarımı (10 ajan) | 2 pilot deney (tam ölçekli soğuk-başlangıç/popülerlik; CROssBAR v1/v2 veri fizibilitesi), 4 rakip makale kurgusu, 3 farklı bakış açılı hakem (kıdemli BiB hakemi, pragmatik PI, klinik farmakolog), 1 sentez. |
| 3. Kilit bulgu doğrulaması (3 ajan) | Önerilen makalenin dayandığı iki pilot bulgunun iki bağımsız sıfırdan uygulaması + benchmark koduna bakan bir şüpheci. |

**Kısıtlar (önemli):**
- Bu bulut ortamından yayıncı siteleri (OUP, Nature, arXiv, bioRxiv, PubMed, IEEE, ACM), DrugBank, KEGG, DDInter, Hugging Face, Google Drive, `crossbar.kansil.org` ve `crossbarv2.hubiodatalab.com` **erişilemezdi**. WebSearch kotası da tükendi.
- Kaynaklar çoğunlukla GitHub üzerinden (yazar kopyaları, README'ler, bib dosyaları, kod) doğrulandı.
- Uydurma kaynak bulunmadı. Ancak bazı **sayılar yalnızca ikincil kaynaklardan** geldi ve Ek A'da işaretlendi. Makalede kullanmadan önce kurum ağınızdan birincil kaynaktan kontrol edilmeli.
- CROssBAR MongoDB'nize (METU) erişim **gerekmedi ve denenmedi**. Tüm analiz repodaki veri anlık görüntüsüyle yapıldı.

---

## 2. Kod ve veri denetimi: bulgular

Her bulgu ikinci bir ajan tarafından kod satırı düzeyinde çürütülmeye çalışıldı. "Doğrulama" sütunu o sonucu gösteriyor.

| # | Bulgu | Yer | Önem (doğrulama sonrası) |
|---|---|---|---|
| F1 | Rastgele sıralı-çift bölmesi: her test ilacı eğitimde de var. Model ilaç kimliğini ve etkileşim eğilimini ezberleyebiliyor. Sadece eğitim etiketlerinden derece skoru AUROC 0.82–0.87 veriyor. | `model_process/data_loader.py:27` | Kritik (doğrulandı) |
| F2 | FFNN çıkışında `nn.Softmax(dim=0)`, yani softmax *batch boyunca* alınıyor. Eşik 0.5/len(X). Bir çiftin tahmini batch'teki diğer çiftlere bağlı; tek çift skorlanınca her şey pozitif çıkıyor. | `model_train/learning_functions.py:42`, `evaluation_functions.py:406` | Majör. Büyük i.i.d. test batch'lerinde etkisi küçük (%0,2–0,9 tahmin değişir), ama dağıtım/yeni ilaç için hata. |
| F3 | "AUC/AUPRC" sert 0/1 tahminlerden hesaplanıyor, eğri alanı değil. Hep-pozitif tahminci "AUPRC" 0.79 (D1) / 0.73 (MP) alıyor. | `model_train/evaluation_functions.py:156` | Majör (doğrulandı; MP FFNN değeri 0.8727 birebir yeniden üretildi) |
| F4 | Tablo 2/3 büyük olasılıkla IQR filtresiyle 179→71 ve 180→73 ilaca inmiş alt kümelerde hesaplanmış. Raporda belirtilmemiş; filtresiz çalıştırmalar MCC 0.28–0.41. | `model_process/data_clean.py:108` + git geçmişi (`872b5e2`) | Kritik (doğrulandı; tablo hücreleri tersine çözülerek test prevalansı 0.61–0.73 bulundu) |
| F5 | node2vec her ego alt-grafında ayrı eğitiliyor. Vektörler esasen benzerlik-ağı derecesini kodluyor (ridge R² ≈ 0.9). p/q ayarı vektörleri %5'ten az değiştiriyor. D1–D2 uzayları hizasız: aynı ilacın kosinüsü 0.08, ilgisiz ilaçlarınki de 0.08. | `featurization/n2v_generation.py:33` | Majör |
| F6 | "Yeni ilaç" (D1→D2) deneyi gerçek soğuk başlangıç değil (46 ortak ilaç) ve özellik uzayları karşılaştırılamaz. Tablo 4/5 satırları ayrıca farklı, filtreli D2 alt kümelerinde hesaplanmış görünüyor. | Tablo 4/5 | Majör |
| F7 | FFNN "isValid" döngüsü doğrulama değil. Her epoch'ta RepeatedKFold ile ~20 kat fazla adım atılıyor ve "en iyi model" referansla saklandığı için hep son model dönüyor. | `learning_functions.py:85–126` | Majör |
| F8 | Hiperparametre ve varyant seçimi test bölmesinde yapılmış. Tek bölme var, seed yok; Tablo 1'deki RF/GB/FFNN farklı rastgele bölmelerde skorlanmış. | `data_loader.py:27` | Majör |
| F9 | HIN-DDI karşılaştırması geçersiz. HIN-DDI: 481 ilaç (yayımlanan veride 406), %50 pozitif, tek 80/20 bölme, sert etiketten AUPR; kodunda sinir ağı yok. | Tablo 1 | Majör |
| F10 | Benzerlik eşiği raporda 0.5, kod ve verilerde 0.3. 0.3'te D1/D2'de her 2-hop ego-grafı tüm grafa eşit oluyor; hop-2 "çöküşünün" nedeni bu. | `data/edgelists/*sim3*` | Majör |
| F11 | D1/D2 `sim_arr` matrisleri repodaki SMILES'tan yeniden üretilemiyor (r = 0.40–0.45). DDI etiket sızıntısı ise **yok** (AUROC 0.58–0.61). | `data/datasets/*/sim_arr.txt` | Minör |
| F12 | Ayna (a,b)/(b,a) ve öz-çiftler (a,a) aynı matriste. Ölçüldü: DDNet skorlarını **şişirmiyor** (ayna-gruplu katlarla sonuçlar aynı). | `featurization/f_mat_generation.py` | Minör |
| F13 | `Normalizer` satır bazlı L2. Yol sayıları (3. uzunlukta 1.000'e kadar) vektör normunu domine ediyor. "Meta-path" adı homojen grafta yanlış; bunlar kesik Katz tarzı yol sayıları. | `data_loader.py:31`, `path_generator.py` | Minör |
| F14 | Embedding'ler yeniden üretilemez: gensim-3 başlatması Python'un süreç-başı hash'ine bağlı ve seed yok. | `n2v_generation.py` | Minör |

---

## 3. Ampirik yeniden değerlendirme

Tüm betikler `analysis/reeval/`, bağımsız doğrulama `analysis/reeval_verify/` altında. İki uygulama katman gürültüsü içinde uyuştu.

### 3.1 Rapordaki protokolle yeniden üretim (AUROC skorlardan)

| Veri | Model | Bizim F1 / MCC | Rapor F1 / MCC | Not |
|---|---|---|---|---|
| MP | RF | 0.82 / 0.66 | 0.53 / 0.26 | Rapordaki RF zayıflığı arama ızgarasından kaynaklanıyor |
| MP | FFNN (orijinal kod) | 0.75 / 0.52 | 0.85 / 0.72 | isValid=True yolunda D1'de MCC 0.50–0.63 (seed'e bağlı) |
| MP | FFNN (sigmoid ile düzeltilmiş) | 0.83 / 0.69 | – | Rapor seviyesine ancak düzeltmeyle ulaşılıyor |
| D1 hop-1 | RF | 0.81 / 0.55 | 0.82 / 0.37 | |
| D1 hop-1 | FFNN düzeltilmiş | 0.87 / 0.71 | 0.90 / 0.71 | |

### 3.2 Aynı protokolde basit kontroller (AUROC / MCC)

| Yöntem | MP | D1 | D2 |
|---|---|---|---|
| Hep-pozitif (F1) | (0.62) | (0.73) | (0.74) |
| Ham benzerlik | 0.61 / 0.14 | 0.57 / 0.11 | 0.58 / 0.16 |
| Yalnız yol özellikleri | 0.64 / 0.22 | 0.69 / 0.23 | 0.70 / 0.28 |
| Derece çarpımı (yalnız eğitim etiketi) | 0.87 / 0.56 | 0.82 / 0.47 | 0.84 / 0.51 |
| One-hot kimlik + RF | 0.86 / 0.55 | 0.81 / 0.45 | 0.84 / 0.51 |
| **DDNet RF** | 0.91 / 0.66 | 0.86 / 0.55 | 0.88 / 0.58 |
| DDNet FFNN (düzeltilmiş) | 0.93 / 0.69 | 0.94 / 0.71 | 0.94 / 0.72 |
| **Kontrol: node2vec yerine rastgele Gauss vektörü + FFNN** | **0.96 / 0.77** | **0.96 / 0.77** | **0.95 / 0.75** |
| **Kontrol: yalnız ilaç kimliği (one-hot) + MLP** | **0.98 / 0.88** | **0.97 / 0.85** | **0.96 / 0.82** |

Yorum: Raporun en iyi sayıları, yoğun ve popülerliğe yanlı bir matriste ilaç kimliğinin ezberlenmesinden geliyor. Not: RF öğrenicisiyle DDNet, rastgele vektörleri sıcak bölmede ~0.05 geçiyor. "Rastgele vektörden kötü" sonucu FFNN'e özgü.

### 3.3 Sızıntısız protokol: sıcak, S1 (bir ilaç yeni), S2 (iki ilaç yeni). AUROC.

| Yöntem | D1 sıcak / S1 / S2 | MP sıcak / S1 / S2 |
|---|---|---|
| DDNet (concat, iki yön, RF) | 0.91 / 0.78 / 0.62 | 0.94 / 0.81 / 0.64 |
| Morgan ECFP4 (aynı RF) | 0.90 / **0.80** / **0.63** | 0.92 / **0.82** / **0.66** |
| Derece (eğitim) | 0.82 / 0.72 / 0.50 | 0.87 / 0.74 / 0.50 |
| Kimlik MLP | 0.96–0.98 / 0.67–0.69 / 0.50 | 0.97–0.98 / 0.71–0.72 / 0.50 |
| SimKNN (benzer ilaçların etiketleri) | 0.76 / 0.70 / 0.61 | 0.79 / 0.74 / 0.65 |
| Ham benzerlik | 0.58 / 0.57 / 0.59 | 0.61 / 0.61 / 0.61 |
| *Tanı: veri kümesi dışı DrugBank derecesi* | *0.81 / 0.81 / 0.79* | *0.84 / 0.84 / 0.84* |

S2 standart sapması 0.03–0.09. Sonuçlar:
- Soğuk başlangıçta DDNet özellikleri Morgan parmak izine eşit, hatta biraz geride.
- Yol özellikleri katkı sağlamıyor.
- Salt "popülerlik" (ilacın veri kümesi dışındaki DrugBank derecesi) S2'de bütün yapı-tabanlı modelleri geçiyor.

### 3.4 Ölçek pilotu: hub seçimi olmadan, 3.618 ilaçlık havuzdan (`analysis/pilot_scale/`)

| Örnek | Pozitif oranı | Derece önseli sıcak / S1 / S2 | ECFP4-HGB sıcak / S1 / S2 | En iyi S2 | Dış derece (tanı) |
|---|---|---|---|---|---|
| U1000 (düzgün) | 0.165 | **0.90** / 0.78 / 0.50 | 0.87 / 0.77 / 0.63 | SimKNN 0.69 | 0.90 (her bölmede) |
| U300 (düzgün) | 0.181 | 0.90 / 0.78 / 0.50 | 0.92 / 0.77 / 0.60 | SimKNN 0.67 | 0.90 |
| H300 (hub, DDNet tarzı) | 0.704 | 0.74 / 0.67 / 0.50 | 0.85 / 0.74 / 0.61 | SimKNN 0.65 | 0.73 |

- **Hub seçimi derece kısayolunu yok etmiyor, gizliyor.** Hub'lar arasında derece varyansı sıkıştığı için "yapı popülerliği geçiyor" gibi görünüyor: sıcakta +0.11. Düzgün örnekte ise bu avantaj +0.015'e iniyor (U300) veya tersine dönüyor (U1000, −0.03).
- Soğuk başlangıç açığının neredeyse tamamı, modelin **yeni bir ilacın ne kadar "etkileşime yatkın" olduğunu tahmin edememesi**.
- Yeni ilaç için mevcut vekiller (hedef sayısı + 4 basit tanımlayıcı) ECFP'nin şans-üstü AUROC'unun %83–93'ünü tek başına veriyor.
- ECFP bu vekillerin üstüne S1/S2'de tutarlı olarak +0.06 ekliyor. Yani küçük ama gerçek bir yapısal sinyal var.

---

### 3.5 "Yeni ilaç" deneyi (D1→D2)

- Rapordaki protokolde tüm DDNet modelleri **AUROC 0.40–0.53**, MCC −0.07…0.03 veriyor.
- Raporun F1 0.64–0.80 değerleri hep-pozitif tahmincinin F1 değerinden (0.736) iyi değil.
- Aynı ilacın D1 ve D2 vektörleri arasındaki kosinüs 0.08, ilgisiz ilaç çiftleriyle aynı. Procrustes analizi yalnızca kısmi ortak geometri gösteriyor.

---

## 4. Literatür: 2022 → 2026'da ne değişti?

### 4.1 Durum özeti

- **Sıcak (rastgele çift) bölmeler doymuş durumda.** DrugBank ikili/çok sınıflı görevlerde yöntemler AUROC ≈ 0.98–0.99 bildiriyor (SSI-DDI, GMPNN-CS, DSN-DDI, R2-DDI, HDN-DDI, KnowDDI). Morgan parmak izi + sığ bir ağ bile rekabetçi (Gil-Sorribes & Molina, J Cheminform 2026; kesin sayılar doğrulanamadı).
- **Alanın gündemi genelleme ve değerlendirme geçerliliği.**
  - **DDI-Ben** (Shen, Zhou, Zhang, Yao; *Bioinformatics* 2025, btaf569): S0/S1/S2, küme bölmeleri ve onay-tarihi ("Real Scene") bölmeleri tanımlıyor ve dağılım kayması altında 10 yöntemin çöktüğünü gösteriyor. DrugBank-86'da en iyi S1 makro-F1 ≈ 57, S2 ≈ 22.5 (ikincil kaynak üzerinden).
  - OpenDDI (arXiv 2026) ve GenRel-DDI (arXiv 2026) hakemsiz. EmerGNN (*Nat Comput Sci* 2023) ve KnowDDI (*Commun Med* 2024) yol/alt-graf tabanlı soğuk-başlangıç yöntemleri.
  - Metin/LLM tabanlı yöntemler (TextDDI EMNLP 2023, ZeroDDI IJCAI 2024, K-Paths KDD 2025, DDI-GPT) kaymaya daha dayanıklı bildiriliyor.
- **Derece/popülerlik kısayolu belgelenmiş bir problem.**
  - Zietz ve ark. (*GigaScience* 2024) XSwap derece-koruyan permütasyon önseli tanımlıyor.
  - Bonner ve ark. (*BiB* 2022): KG embedding skorları dereceyi izliyor.
  - Aiyappa ve ark. (*ICML* 2025): bağlantı tahmininde derece yanlılığı. Çalışma bir ilaç etkileşim ağı da içeriyor.
  - Öncüller: AI-Bind (*Nat Commun* 2023, DTI), Guney (PSB 2017), Park & Marcotte (*Nat Methods* 2012), Kpanou ve ark. (*BMC Bioinf* 2021), Shtar ve ark. (*PLOS ONE* 2019).
  - Dolayısıyla "derece kısayolu var" demek tek başına özgün değil. Özgün olabilecek kısım DDI'ye ve soğuk başlangıca özgü nicelendirme.
- **ogbl-ddi** (OGB, NeurIPS 2020) DDNet'in formülasyonunun standart benchmark'ı: homojen DrugBank DDI grafı, 4.267 ilaç, 1,33 M kenar, protein-hedef bölmesi, Hits@20, CC-0 lisansı. Liderlik tablosunda node2vec ≈ 0.23, en iyiler ≈ 0.95+.

### 4.2 Özgünlük: DDNet'in bileşenleri

| Bileşen | En yakın önceki çalışma | Durum |
|---|---|---|
| Parmak izi benzerliğinden DDI | Vilar 2012 (JAMIA), Vilar 2014 (Nat Protoc), INDI 2012, Zhang 2017, NDD 2019 (Sci Rep) | Özgün değil |
| Homojen benzerlik grafı + gömme + sınıflandırıcı | Bumgardner, Tanvir, Saifuddin & Akbas, IEEE BigData 2021 (HIN-DDI grubu); HAN-DDI'nin "HG3" ve node2vec temel çizgileri (BioKDD 2022) | Özgün değil |
| Ego/yerel alt-graf | WLNM 2017, SEAL 2018, GraIL 2020, G-Meta 2020, SumGNN 2021, KnowDDI 2024, CSSE-DDI (NeurIPS 2024) | "Her ego-grafa ayrı node2vec" hiçbir yerde yok, ama yöntemsel olarak kusurlu (hizasız uzaylar) |
| Yol-çarpımı "meta-path" | DASPfind 2016, yerel yol indeksi (Lü 2009), Katz 1953; meta-path DDI: HIN-DDI 2021, TCBB 2024 dergi sürümü, MP-DDI (BiB 2023), BIBM 2022 | Özgün değil |

**Olumlu bir not:** EmerGNN'de HIN-DDI (meta-path özellikleri), TWOSIDES yeni-ilaç ayarında en güçlü temel çizgiydi. Yani *heterojen* KG'lerde yol özellikleri yeni ilaçlara iyi aktarılabiliyor. DDNet'te aktarılmamasının nedeni grafın homojen olması ve yalnızca kimyasal benzerlikten kurulması.

**Ön yayın/intihal kontrolü:** DDNet raporu arXiv/bioRxiv/ResearchGate'te bulunamadı. Aynı yöntemi yayımlayan kimse de bulunamadı. Bu nedenle ön yayın çakışması yok.

### 4.3 2026'da bir DDI makalesinden beklenenler (hakem kontrol listesi)

- Tekrarsız sırasız çiftler ve ilaç-ayrık S1/S2 bölmeleri. Ek olarak küme (iskelet) ve onay-tarihi bölmeleri.
- Gerçekçi prevalans veya dereceyle eşleştirilmiş negatifler. "Listede olmayan" çift gerçek negatif değildir (pozitif-etiketsiz problemi).
- Zorunlu temel çizgiler: hep-pozitif, derece/XSwap önseli, kimlik, ham benzerlik / Vilar / SimKNN, Morgan-MLP, en az bir güncel soğuk-başlangıç yöntemi (EmerGNN / DDI-Ben takımı).
- Metrikler skorlardan hesaplanmalı: AUROC ve ortalama kesinlik (AP). Eşik yalnızca doğrulama kümesinde seçilmeli. ≥5 seed ve güven aralığı raporlanmalı.
- DOME kontrol listesi, kod + veri yayını ve lisans uyumu.

### 4.4 CROssBAR'ın durumu

- **CROssBAR v1** (*NAR* 2021; Ahmet Atakan ortak yazar): CC BY 4.0. GitHub README'si hâlâ artık kapalı olan EBI API'sini gösteriyor.
  - Açık v1 KG'lerinde DDI, yan etki ve enzim/taşıyıcı verisi **yok**. Yalnızca ilaç kimlikleri, SMILES, hedefler, ChEMBL aktiviteleri, PPI, hastalık ve HPO var.
  - METU'daki MongoDB'nin şeması açık kodda belgelenmemiş. İlaç dokümanlarının enzim, taşıyıcı, ATC veya `drug_interactions` alanları içerip içermediği sunucuda kontrol edilmeli.
- **CROssBARv2** (bioRxiv 2026, doi 10.64898/2026.04.12.718028):
  - Yazarlar: Şen, Ulusoy, Darcan, Ergün, Lobentanzer, Rifaioğlu, Türei, Saez-Rodriguez, Doğan (HUBioDataLab). **İki DDNet yazarı da bu çalışmada yok.**
  - İçerik: BioCypher/Neo4j tabanlı, ~2,71 M düğüm / ~12,69 M ilişki, 14 düğüm tipi, 51 kenar tipi, 34 kaynak. SELFormer ilaç embedding'leri, CROssBAR-LLM, GraphQL API ve PyG dışa aktarma betiği var.
  - **DDI kenarları:** DDInter (şiddet: Major/Moderate/Minor/Unknown + aksiyon; büyük olasılıkla DDInter v1) ve KEGG (kontrendikasyon/önlem). DrugBank DDI kullanılmıyor.
  - Eksikler: DrugBank enzim/taşıyıcı/carrier yok (farmakokinetik sinyal eksik), onay tarihi yok, veri lisansı belirtilmemiş.
  - Önemli bir sızıntı riski: varsayılan PyG dışa aktarımı DDI kenarlarını içeriyor. KG üzerinde model eğitenler test etiketlerini görebilir.
  - Özetinde bir DDI benchmark'ı görünmüyor. Bu bir fırsat, ama HUBioDataLab ile koordinasyon gerektirir.

### 4.5 Mecralar ve maliyet (OpenAPC medyan APC, 2023+)

| Katman | Mecralar |
|---|---|
| Mevcut haliyle | Yalnızca ön yayın veya düşük çıtalı bildiri. **Önerilmez.** |
| Titiz değerlendirme / negatif sonuç | *BiB* (~€3,4k), *Bioinformatics* (~€3,6k), *GigaScience* (~€2,7k), *Bioinformatics Advances* / *BMC Bioinformatics* (~€2,4k), *J Cheminform* (~€1,9k); ACM BCB (~%29 kabul), IEEE BIBM (~%20) |
| Güçlü soğuk-başlangıç sonuçlu yeni yöntem | *BiB*, *Bioinformatics*, ISMB/ECCB, JCIM, TCBB, JBHI. KDD/NeurIPS yalnızca SOTA'yı açıkça geçerse. |
| APC'siz yol | Hibrit dergiler (JCIM, Comput Biol Med, TCBB, JBHI) abonelik yoluyla |

---

## 5. Karar

**Mevcut haliyle yayımlamayın.** Bir hakem ilk turda F1/F4/F9'u ve derece temel çizgisinin eksikliğini yakalar. CROssBAR'a yakın hakemlerde itibar riski de yaratır.

2022 çalışmasının 2026'daki gerçek değeri başka bir yerde: kendi başına **iyi belgelenmiş bir "kısayol anatomisi" örneği**. Bir değerlendirme makalesinin motive edici vaka analizi olarak (Kutu 1) kullanılabilir.

---

## 6. Değerlendirilen makale kurguları ve hakem puanları

Dört kurgu bağımsız ajanlar tarafından tasarlandı ve üç farklı bakış açılı hakem tarafından puanlandı. Her boyut 1–10; toplam 50 üzerinden, üç hakemin ortalaması.

| Kurgu | Özet | Özgünlük | Yayınlanabilirlik | Fizibilite | Toplam | Karar |
|---|---|---|---|---|---|---|
| **A. Değerlendirme bilimi** | Soğuk-başlangıç DDI benchmark'ları ilaç düzeyi ana etkileri mi ölçüyor? Eğitimsiz "taban" çizgiler, eklemeli kontroller, sıralama değişimi. | 5 | 6 | 6–7 | **30.0** | **Seçildi (omurga)** |
| D. Minimal düzeltilmiş DDNet | Popülerlik kontrollü yeniden değerlendirme, küçük bildiri | 2–3 | 6–7 | 9 | 28.7 | Kutu 1 + sigorta planı |
| C. Yöntem (MEP-DDI) | KG ile tahmin edilen ilaç eğilimleri + yol artığı | 4–5 | 4–5 | 4 | 22.3 | Sadece "eklemeli kontrol" ve hedef-kNN parçası alındı |
| B. CROssBAR şiddet/mekanizma benchmark'ı | CROssBARv2 üzerinde sızıntı ve derece kontrollü soğuk-başlangıç benchmark'ı | 4–6 | 4–5 | 3 | 21.7 | 450–520 saat, 6 dış bağımlılık. Opsiyonel küçük modül. |

Neden B ana makale değil:
- HUBioDataLab onayı gerekiyor. Bu grup aynı zamanda fikri kapma riski en yüksek olan grup.
- DDInter lisansı belirsiz. KG dökümü bu ortamdan erişilemedi ve veri lisansı yok.
- v2'de enzim/taşıyıcı verisi yok. DrugBank kimlik bilgisi gerekiyor.
- K-Paths, PKAG-DDI ve MARD'a göre özgünlüğü ince.

Ayrıca B'nin DDInter pilotu yan bulgular üretti:
- Uygun DDInter çiftlerinin %70–79'u zaten DrugBank DDI tablosunda. DrugBank'te eğitilen modeller "soğuk" test çiftlerinin çoğunu görmüş oluyor.
- Varsayılan CROssBARv2 PyG dışa aktarımı şiddet etiketli DDI kenarlarını içeriyor.
- Bu bulgular HUBioDataLab'a nezaket notu olarak iletilebilir.

---

## 7. Önerilen makale

**Çalışma başlığı:** *What do emerging-drug DDI benchmarks measure? Drug-level propensities versus pair-specific interaction*

Alternatif başlıklar:
- *Perpetrator, victim or pair? Additive controls for emerging-drug interaction prediction*
- *Mostly main effects: decomposing cold-start drug–drug interaction performance on DDI-Ben*

**Ana fikir.** Bir DDI tipi skoru şöyle ayrıştırılabilir:

logit_r(a,b) = μ_r + α_r(a) + β_r(b) + γ_r(a,b)

Burada α ve β ilaç düzeyi eğilimlerdir. Farmakolojide bunlar "perpetrator/victim" (örn. güçlü CYP3A inhibitörü, dar terapötik indeksli substrat) ve PD sınıf yükümlülükleridir (QT, serotonerjik, kanama). γ ise çifte özgü mekanizma eşleşmesidir.

**İddia:** Yayımlanmış soğuk-başlangıç (S1/S2) skorlarının büyük kısmı α+β'den geliyor. Bu bir "sızıntı" değil, meşru farmakoloji. Ama benchmark'lar bunu ayrı raporlamazsa "yeni ilaç için etkileşim tahmini" iddiası şişkin kalır.

### 7.1 Kilit pilot bulgular (bağımsız doğrulama ayrıntıları §8.1'de)

1. **TWOSIDES, DDI-Ben rastgele S1 (birebir doğrulandı).**
   - Hiç eğitim yok: yalnızca *bilinen* ilacın o yan-etki etiketi için eğitim grafındaki derecesi kullanıldı.
   - Sonuç, benchmark'ın kendi değerlendiricisiyle: ROC-AUC **87.5** / PR-AUC **83.6**. Hakemsiz DRIFT modeli 82.3 / 80.3 bildiriyor.
   - Mekanizma: S1 negatifleri pozitiflerden bağımsız rastgele (bilinen, yeni) çiftler. Pozitiflerde bilinen ilaç popüler ilaçlardan geliyor, negatiflerde neredeyse düzgün seçiliyor. Derece skoru bu farkı yakalıyor.
   - Düzeltme: bilinen ilacı ve etiketi koruyup yalnızca yeni ilacı değiştiren "partner-eşleşmeli negatifler". Bunlarla aynı skor tam 50'ye düşüyor.
   - Negatiflerin %17'si ters yönde kayıtlı pozitif çiftler (etiket gürültüsü).
2. **DrugBank-86, DDI-Ben rastgele bölme (kısmen doğrulandı; sayı tarife bağlı).**
   - Eğitimsiz, çift terimi olmayan eklemeli bir "ana etki" kuralı kullanıldı: bilinen ilacın rol-özgü tip dağılımı × yeni ilacın 10 en benzer (Tanimoto) komşusundan türetilen dağılım.
   - Sonuç: S1 makro-F1 **≈54–57** (tarife göre; acc 65–68, κ 58–61), S2 **≈16–22**.
   - Yayımlanmış en iyi değerler S1 ≈ 57 (EmerGNN, DDI-GPT, TextDDI), S2 ≈ 22.5. Bu değerler ikincil kaynaktan ve teyit edilmeli.
3. **Kümeli ve onay-tarihi bölmeleri** ana etkilere daha dirençli.
   - Bu bölmelerde CROssBAR v1 hedef listelerinden hesaplanan hedef-kNN eğilimleri, Tanimoto-kNN'e göre makro-F1'i belirgin artırıyor: küme S1 28.8 → 41.2 (test), 26.7 → 34.6 (doğrulama).
   - Kappa'da artış küçük. Hedef listeleri DrugBank'ten türetildiği için "küratörlük bağlantısı" kontrolü gerekiyor (yalnız ChEMBL biyoaktivite varyantı).

### 7.2 Makale iskeleti (BiB araştırma makalesi, ~7–9 bin kelime)

1. **Giriş:** S0/S1/S2 protokolü. Skorlar iki şeyi karıştırıyor: ilaç düzeyi yükümlülük ve çifte özgü mekanizma.
2. **Çerçeve:**
   - Ayrıştırma ve farmakolojik gerekçe.
   - Üç kontrol: (a) eğitimsiz tabanlar, (b) aynı kodlayıcıyla eğitilmiş, çift terimi olmayan eklemeli model, (c) herhangi bir modelin logit yüzeyinin iki-yönlü ANOVA ile eklemeli izdüşümü.
   - Tutma oranı: R = (eklemeli − çoğunluk) / (tam − çoğunluk).
   - Kontroller önce ekilmiş etkileşim payı %0/10/30 olan sentetik verilerde doğrulanır.
3. **Veri ve yöntemler:**
   - DDI-Bench (commit `dfbeeab`): DrugBank-86 rastgele/küme, TWOSIDES rastgele/küme, Real Scene. Resmî bölme + 3 ilaç-ayrık yeniden bölme.
   - Temel çizgi merdiveni: çoğunluk, partner önseli, ME-kNN, derece, XSwap, Vilar, SimKNN, ATC/sınıf-çifti kuralı, hedef-kNN (CROssBAR v1) ve yalnız ChEMBL varyantı.
   - Yeniden çalıştırılacak modeller (3 seed): DDI-Ben MLP, DRIFT benzeri Morgan-MLP, SSI-DDI, EmerGNN. G2'ye göre +TIGER.
   - İstatistik: ilaç düzeyi küme bootstrap, TOST eşdeğerlik (S1 ±2, S2 ±3 makro-F1), ön-kayıtlı eşikler.
4. **Sonuçlar:**
   - TWOSIDES negatif-kurulum kısayolu ve partner-eşleşmeli negatifler.
   - DrugBank tabanları ve modeller.
   - Modeller ne kadar eklemeli?
   - Sıralamalar değişiyor mu (Kendall τ)?
   - Etiket şablonlarının ne kadarı tek ilacın sınıfıyla belirleniyor (PK↑/PK↓/PD-toksisite/etkinlik kaybı gruplaması)?
   - Ana etkilerin bittiği yer: küme/Real Scene ve hedef eğilimleri.
5. **Kutu 1 — "Bir kısayolun anatomisi":** 2022 DDNet öz-denetimi.
   - Hub seçimi: prevalans 0.70'e karşı 0.17; AP kaldıracı ~4× → 1.3×.
   - Kimlik MLP 0.96–0.98. D1→D2 şansa eşit.
   - HIN-DDI yayınındaki veri de benzer: %43 pozitif, 81,5. derece yüzdeliği.
6. **Tartışma ve öneriler:** Zorunlu tabanlar, partner-eşleşmeli negatifler, küme/onay-tarihi bölmelerinin birincil olması, R raporlama. Kısıtlar: şablon etiketler; TWOSIDES'ın FAERS sinyali olması; listelenmemiş ≠ negatif.
7. **Erişilebilirlik:** Yeni adla temiz bir repo (MIT/Apache), Zenodo DOI, DOME kontrol listesi. DrugBank, KEGG ve DDInter içeriği yayınlanmaz.

### 7.3 Yol haritası (iki yarı zamanlı yazar, haftada toplam ~20 saat, 28 Eylül 2026'dan itibaren)

Sorumlular: **AA** = A. Atakan (tabanlar, CROssBAR, istatistik, baş yazar). **ASO** = A. S. Özdilek (ortamlar, yeniden çalıştırmalar, GPU, model kancaları, yeniden bölmeler).

| Evre | Haftalar | İçerik | Saat (AA/ASO) | Kapı |
|---|---|---|---|---|
| 0. Doğrulama ve kurulum | H1–2 | AA: DDI-Ben tam metni ve Tablo 9'u kurum ağından teyit; DDI-Ben yazarlarına e-posta (lisans, silinmiş `train_1.txt`, bölme üreteci, tahmin dosyaları); METU Mongo envanteri. ASO: yeni adla repo, ortamlar, zamanlı duman testleri, pilot sonuçlarının birebir yeniden üretimi. | 16/16 | G0 |
| 1. Tabanlar, kontroller, ilk yeniden çalıştırmalar | H3–6 | Taban kütüphanesi, bootstrap/TOST, sentetik doğrulama, ön-kayıt (git tag). Yeniden bölme ve partner-eşleşmeli negatif üreteçleri, eklemeli başlıklar, GPU kuyruğu. | 40/44 | G1, G2 |
| 2. Ayrıştırma ve yapıcı kısım | H7–10 | İzdüşüm analizi, R tabloları, Kendall τ, etiket-şablon analizi, hedef-kNN (küme/Real Scene). Çıkarım kancaları, küme bölmesi çalıştırmaları. | 44/40 | G3 (belirleyici), G4 |
| 3. Ön yayın v1 | H11–12 | Yöntem ve Sonuçlar, Kutu 1, şekiller, özgünlük taraması. Etiketli sürüm v0.1. | 26/16 | G5: **20 Aralık'a kadar ön yayın** |
| 4. Tamamlama | H13–16 | Tartışma, DOME, opsiyonel DDInter modülü ve 2×2 CYP testi. | 30/26 | G6 |
| 5. Kırmızı takım ve gönderim | H17–18 | İddia denetimi (TOST, iki metrik), kapak mektubu, lisans izinleri. | 18/12 | G7 |
| Tampon | H19–20 | | 10/10 | **Gönderim ≈ 5 Şubat 2027** |

Toplam ≈ 350 kişi-saat, revizyon için ek 40–60 saat. GPU ≈ 150–250 saat (24 GB tek kart, doğrulanmamış tahmin; EmerGNN baskın). Tabanlar CPU'da dakikalar sürüyor.

### 7.4 Git/geçme kapıları (özet)

- **G0 (H2):** Bu ortamda üretilen tüm pilot "çapa" sayıları sizin makinenizde ±0.1 içinde tekrar üretilmeli. DDI-Ben Tablo 9 yayımlanmış makaleden teyit edilmeli. Teyit edilemezse "yayımlanmış SOTA'yla eşleşiyor" iddiası kullanılmaz.
- **G1 (H4):** TWOSIDES kısayolu 4 rastgele bölmenin hepsinde S1 ROC-AUC ≥ 80 kalmalı; aksi halde tek paragrafa iner. ME-S1 ortalaması ≥ 52 ve S2 ≥ 18 olmalı; aksi halde yayımlanmış sayılarla kıyas çıkarılır.
- **G2 (H6):** {DDI-Ben MLP, SSI-DDI, TIGER, EmerGNN} içinden ≥ 3'ü raporlanan S1 değerini max(3, 1,5×SD) içinde üretmeli. **Durdurma ölçütü (H8):** ≤ 1 MLP-dışı model çalışıyorsa VE TWOSIDES bulgusu G1'de düştüyse, sigorta planına (kurgu D, ~100 saat, ACM BCB / SIU) geçilir.
- **G3 (H9, belirleyici, ön-kayıtlı):**
  - Medyan R ≥ 0.80 (S1) ve ≥ 0.70 (S2), hem makro-F1 hem κ'da → "çoğunlukla ilaç düzeyi eğilim" manşeti.
  - Değilse, bir model kendi eklemeli izdüşümünü S1'de ≥ 3 / S2'de ≥ 5 makro-F1 aşıyorsa (GA'sı 0'ı dışlayarak) → "hangi modeller çifte özgü sinyal öğreniyor" çerçevesi.
  - İkisi de değilse → nötr "Ayrıştırma" başlığı, bir basamak aşağı mecra.
- **G4 (H10):** Hedef-kNN, küme S1'de Tanimoto-kNN'e göre ≥ +3 makro-F1, κ ≥ 0 ve 4 bölmenin ≥ 3'ünde iyileşme sağlamalı. Yalnız ChEMBL varyantı kazancın ≥ %50'sini korumalı.
- **G5 (H11):** Tam web özgünlük taraması. Birisi aynı tabanları veya TWOSIDES artefaktını yayımladıysa "bağımsız replikasyon + ayrıştırma yöntemi" olarak yeniden çerçevelenir ve hedef GigaScience / Bioinformatics Advances olur.

### 7.5 Mecra merdiveni

1. *Briefings in Bioinformatics* (birincil)
   - Alternatif: *Bioinformatics* (DDI-Ben'in yayımlandığı yer; DDI-Ben yazarları hakem olabilir).
2. *GigaScience* (Zietz 2024 burada yayımlandı)
3. *Bioinformatics Advances* / *BMC Bioinformatics* (minimum yayımlanabilir birim)
4. APC yoksa: JCIM / TCBB (abonelik yolu)

Buna paralel olarak 11.–12. haftada bioRxiv/arXiv ön yayını yapılması öneriliyor (öncelik için).

Sentezin kaba yargısı (veri değil): ~12 ayda merdivenin bir yerinde hakemli yayın olasılığı %80–85, BiB'de ilk denemede kabul olasılığı %35–50.

---

## 8. Doğrulama notları

### 8.1 Kilit pilot bulguların bağımsız doğrulaması

Üç ajan kullanıldı: iki sıfırdan yeniden uygulama (önce pilot koduna bakmadan) ve benchmark değerlendirici koduyla satır satır karşılaştırma yapan bir şüpheci. Betikler `analysis/ddiben_pilots/verify_*` altında. Karar: **"iddialar uyarılarla geçerli"**. Açıklanamayan veya diskalifiye edici bir hata bulunmadı ve test etiketi sızıntısı yok.

**Bulgu 2 (TWOSIDES kısayolu): birebir doğrulandı ve en sağlam manşet bu.**
- DDI-Bench'in kendi `trainer.py` değerlendiricisinin birebir kopyasıyla ve EmerGNN değerlendiricisiyle test S1 ROC-AUC **87.52** / PR-AUC **83.58**; doğrulama S1'de 88.03 / 84.38.
- Aynı taban kümeli bölmede test S1 85.5 / 81.6, S0'da 95.2 / 93.7 veriyor. S2'de tanım gereği 50.
- **Mekanizma düzeltildi.** Negatifler "bilinen ilacı değiştirilmiş" çiftler değil. 9.890 negatiften 9.761'i pozitifle **hiçbir ilacı paylaşmayan** rastgele (bilinen, yeni) çiftler.
  - Pozitiflerdeki bilinen ilaç dereceyle orantılı (Spearman +0.98).
  - Negatiflerdeki bilinen ilaç 516 eğitim ilacına neredeyse düzgün dağılmış (Spearman −0.57).
  - Sonuç olarak bilinen ilacın ortalama eğitim derecesi pozitiflerde 185, negatiflerde 103. Derece skoru ikisini bu yüzden ayırıyor.
- **Partner-eşleşmeli negatiflerle** (bilinen ilaç ve etiket sabit, yalnız yeni ilaç değişiyor) aynı skor tam olarak **50.0 / 50.0**'a düşüyor. Benchmark AUROC'unun neredeyse tamamı partnerler-arası sıralamadan geliyor; aynı-etiket karşılaştırmalarının yalnız %0,16'sı partner-içi.
- Ek bulgu: test S1 negatiflerinin **%17'si** (1.684/9.890) aynı veri kümesinde ters yönde pozitif olarak kayıtlı. Bunların %7,1'i değerlendirilen etiketi de paylaşıyor. Yani negatiflerde etiket gürültüsü var.
- Karşılaştırma: hakemsiz DRIFT modeli aynı protokolde 82.3 / 80.3 bildiriyor. Protokol tutarlı, ama DDI-Ben'in kendi TWOSIDES tablosuna erişilemedi.
- Erişilebilir hiçbir kaynakta bu artefakt raporlanmamış. OpenDDI'nin tam metni okunamadığı için bu kesin değil.

**Bulgu 1 (DrugBank-86 ana-etki tabanı): kısmen doğrulandı. Sayı tarife bağlı, sıralama değil.**
- Pilotun 57.0 / 67.0 / 60.0 (S1) ve 22.3 / 40.5 / 23.5 (S2) değerleri birebir üretilebiliyor, ama yalnızca belirli bir tarifle:
  - dağılımların öncele doğru büzülmesi: (sayım + önsel) / (n + 1),
  - yeni ilaç için komşu *sayımlarının* benzerlik-ağırlıklı ortalaması.
- İddianın düz ifadesiyle (ham dağılımlar, 10 komşunun ağırlıksız ortalaması) sonuç S1 **53–55**, S2 **16–18** makro-F1.
- 54 makul varyant taramasında S1 53.2–57.4 ve S2 14.7–25.3. Doğrulama kümesinde seçilen varyant testte 54.1 veriyor. Pilot tarifinde k doğrulamada seçilirse (5 veya 20) testte 56.8–58.5.
- Doğruluk ve κ sağlam: S1'de 65–68 / 58–61.
- Bootstrap %95 GA: satır bazlı [54.4, 59.3], yeni-ilaç kümeli [51.0, 62.2].
- **Karşılaştırılan yayımlanmış sayılar doğrulanamadı.** DDI-Ben Tablo 9'a yalnızca hakemsiz DRIFT'in kopyasından ulaşıldı. Bu kopyada muhtemel bir aktarım hatası var: DDI-GPT'nin S2 değeri SAGAN'ınkiyle birebir aynı.
- Değerlendirici farkı: DDI-Ben `trainer.py` etiket için tüm dosyalardaki ilişkilerin en küçük indeksini alıyor ve son eksik batch'i atıyor. Pilot ise EmerGNN/TextDDI kuralını kullandı. Etki ≤ 0,4 puan.
- DRIFT ön yayını S0 için eğitimsiz ilaç-başı arama tablosu bildiriyor (%78,4 doğruluk). Ama "S1/S2 aktarılabilir etkileşim örüntülerini ölçer" diye savunuyor ve S1/S2 tabanı raporlamıyor. Önerilen makale doğrudan bu iddiayı test ediyor.

**Plana etkisi:**
- **TWOSIDES bulgusu makalenin açılış sonucu olmalı.**
- DrugBank iddiası "eğitimsiz eklemeli bir kural yayımlanmış en iyi S1 skorunun birkaç puan yakınında (≈54–57'ye karşı ≈57)" diye ifade edilmeli, "eşit veya üstün" değil. Tarif doğrulama kümesinde seçilip ön-kayıtla sabitlenmeli.
- G1 eşiği (ME-S1 ≥ 52) düz tarifle bile geçiliyor.
- "Yayımlanmış sayıyla eşleşiyor" cümlesi G0'da DDI-Ben tam metninden teyit edilmeden kullanılmamalı.

### 8.2 Birinci tur doğrulayıcılarının düzelttiği noktalar

- Ayna ve öz-çift tekrarı DDNet skorlarını **şişirmiyor**. Asıl etkenler sıcak bölme, kimlik/popülerlik ezberi, pozitif-çoğunluk ve sert-etiket AUPRC. (Önemsiz bir "ayna-kopyala" kuralı F1 0.93 alıyor, ama DDNet bunu kullanmıyor.)
- Orijinal FFNN kodu isValid=True yolunda D1'de MCC 0.50–0.63'e ulaşıyor (seed'e bağlı). "Hiç üretilemiyor" demek abartı olur; rapordaki 0.71'in altında ve kararsız.
- RF öğrenicisiyle DDNet, rastgele vektör kontrolünü sıcak bölmede ~0.05 ve dereceyi S1'de 0.06–0.07 geçiyor. Morgan'a eşitlik sonucu ise değişmiyor.
- "Dış DrugBank derecesi" tanısı sızıntı değil: yalnız veri kümesi dışı partnerlerle hesaplanınca da aynı sonucu veriyor. Ama yeni bir ilaç için bu bilgi mevcut değil.

---

## 9. Hemen yapılması gerekenler (makaleden bağımsız)

1. **Lisans temizliği (karar sizde; ben dokunmadım).** Herkese açık repo şunları içeriyor:
   - `data/drug-drug_interaction_Drugbank.csv` (2,77 M satır DrugBank DDI),
   - `data/drugs.json` (DrugBank kimlikleri, SMILES, hedefler),
   - KEGG SIMCOMP'tan türetilmiş benzerlikler.

   Seçenekler: (a) dosyaları kaldırmak (git geçmişini yeniden yazıp yazmamak sizin kararınız), (b) repoyu gizli yapmak, (c) makalede bu repoya hiç bağlantı vermeyip yeni ve temiz bir repo açmak.
2. **İsim.** "DDNet" (IEEE TMI 2018 ve diğerleri) ile "DDINet" (Sci Rep 2025) dolu. Yeni projeye çakışma kontrolü yapılmış yeni bir ad verin.
3. **README.** CROssBAR veri kaynağı olarak artık çalışmayan EBI API'sini gösteriyor. Güncel kaynak olarak METU MongoDB veya CROssBARv2 belirtilmeli.

---

## 10. Sizden gerekenler

- **Plan kararı ve kapasite:** Tek makale planı (A omurgası + C/D parçaları + B'den opsiyonel modül). Kişi başı haftada 8–12 saat, Şubat 2027'ye kadar. Baş yazar kim olacak?
- **METU CROssBAR v1 MongoDB:** **Şu an gerekmiyor.** 1. haftada lazım olacak. Salt-okunur erişim (VPN/SSH tüneli) ve şu bilgiler gerekli:
  - koleksiyon listesi ve belge sayıları,
  - `drugs`, `activities` ve `targets` için birer örnek belge,
  - ilaç belgelerinde enzim, taşıyıcı, ATC, onay veya `drug_interactions` alanları olup olmadığı,
  - derleme tarihi ve kaynak sürümleri.

  Kimlik bilgilerini git'e ve bu bulut oturumuna koymayın.
- **DrugBank akademik lisansı** (her yazar için). Kutu 1 sayıları lisanslı indirmeden yeniden türetilmeli.
- **Kurum ağından tam metin erişimi:**
  - DDI-Ben makalesi ve eki (Tablo 9 ve TWOSIDES tablosu),
  - OpenDDI,
  - Kpanou 2021,
  - Guney 2017,
  - EmerGNN ek dosyası,
  - IC-index (arXiv 2510.14419),
  - AI-Bind.
- **DDI-Bench yazarlarına e-posta onayı** (Y. Zhang, Q. Yao). Repoda lisans dosyası yok.
- **Donanım ve bütçe:** GPU modeli ve VRAM, çok günlük işlerin gözetimsiz çalışıp çalışamayacağı, APC bütçesi (BiB ≈ €3,4k). ODTÜ/ULAKBİM'in OUP ile oku-yayımla anlaşması var mı (doğrulanmadı)?
- **HUBioDataLab ile ilişki (opsiyonel):** CROssBARv2'deki DDI kenarı ve PyG sızıntı notunu paylaşmak, ileride B kurgusu için işbirliği.

---

## Ek A. Temel kaynaklar ve doğrulama durumu

✔ = varlığı ve ilgili iddia birincil veya yazar kopyasından doğrulandı. ◐ = varlığı doğrulandı, belirtilen sayı ikincil kaynaktan geldi veya doğrulanamadı. ⚠ = hakemsiz ön yayın.

**DDNet'in karşılaştırma ve öncül çalışmaları**
- ✔ Tanvir, Islam, Akbas. *Predicting DDIs using meta-path based similarities (HIN-DDI).* IEEE CIBCB 2021. doi:10.1109/CIBCB49929.2021.9562802. Veri: github.com/farhantanvir1/HIN-DDI (406 ilaç, %43,1 pozitif). MP kümesi bunun 289 ilaçlık alt kümesi.
- ◐ Tanvir, Saifuddin, Islam, Akbas. *DDI Prediction With HIN – Meta-Path Based Approach.* IEEE/ACM TCBB 2024. doi:10.1109/TCBB.2024.3417715 (HIN-DDI dergi sürümü; sayılar okunamadı).
- ✔ Tanvir, Saifuddin, Akbas. *HAN-DDI.* BioKDD 2022 (arXiv 2207.05672). 513 ilaç, mevcut ilaçlarda F1 95.18, yeni ilaçlarda 82.87.
- ✔ Bumgardner, Tanvir, Saifuddin, Akbas. *DDI Prediction: a Purely SMILES Based Approach.* IEEE BigData 2021. doi:10.1109/BigData52589.2021.9671766.
- ✔ Saifuddin ve ark. *HyGNN.* ICDE 2023 (arXiv 2206.12747).
- ✔ Vilar ve ark. JAMIA 2012; Nat Protoc 2014 (doi:10.1038/nprot.2014.151).
- ◐ Rohani & Eslahchi. *NDD.* Sci Rep 2019. doi:10.1038/s41598-019-50121-3.
- ✔ Celebi ve ark. BMC Bioinformatics 2019. doi:10.1186/s12859-019-3284-5.
- ✔ Ba-alawi ve ark. *DASPfind.* J Cheminform 2016.

**Yöntemler ve benchmark'lar**
- ✔ Zhang ve ark. *EmerGNN.* Nat Comput Sci 2023. doi:10.1038/s43588-023-00558-4.
- ◐ Wang, Yang, Yao. *KnowDDI.* Commun Med 2024. doi:10.1038/s43856-024-00486-y.
- ✔ Du ve ark. *CSSE-DDI.* NeurIPS 2024 (arXiv 2411.01535).
- ◐ Yu ve ark. *SumGNN.* Bioinformatics 2021.
- ◐ Zhu ve ark. *TextDDI.* EMNLP 2023. ◐ *ZeroDDI.* IJCAI 2024. ◐ *K-Paths.* KDD 2025.
- ◐ Shen, Zhou, Zhang, Yao. *DDI-Ben.* Bioinformatics 41(11) btaf569, 2025. Kod ve veri: github.com/LARS-research/DDI-Bench. Tablo 9 sayıları ikincil kaynaktan geldi.
- ✔ Hu ve ark. *OGB (ogbl-ddi).* NeurIPS 2020.
- ⚠ OpenDDI (arXiv 2602.00539), GenRel-DDI (arXiv 2601.15771), DRIFT (GitHub), Anti-DDI (GitHub; önceki makalesi geri çekilmiş).
- ◐ Gil-Sorribes & Molina. J Cheminform 2026. doi:10.1186/s13321-025-01128-8. "AUROC 99.4" değeri doğrulanamadı.

**Değerlendirme ve yanlılık**
- ✔ Park & Marcotte. Nat Methods 2012. doi:10.1038/nmeth.2259.
- ✔ Dewulf, Stock, De Baets. *Cold-start problems…* Pharmaceuticals 14(5):429, 2021.
- ◐ Kpanou ve ark. BMC Bioinformatics 22:477, 2021.
- ◐ Guney. PSB 2017.
- ◐ Shtar ve ark. PLOS ONE 2019.
- ✔ Zietz ve ark. GigaScience 2024. doi:10.1093/gigascience/giae001.
- ✔ Bonner ve ark. BiB 2022. doi:10.1093/bib/bbac279.
- ✔ Aiyappa, Wang, Kim, Seckin, Ahn, Kojaku. ICML 2025 (PMLR 267:874–908).
- ◐ Li ve ark. *HeaRT.* NeurIPS 2023 D&B.
- ✔ Walsh ve ark. *DOME.* Nat Methods 2021.
- ✔ Schumacher ve ark. *Effects of randomness on node embeddings.* ECML-PKDD 2021 GEM çalıştayı (arXiv 2005.10039).

**Kaynaklar ve veri**
- ✔ Doğan, Ataş, Joshi, Atakan ve ark. *CROssBAR.* NAR 49(16):e96, 2021. doi:10.1093/nar/gkab543.
- ✔ Şen ve ark. *CROssBARv2.* bioRxiv 2026. doi:10.64898/2026.04.12.718028. Kod: github.com/HUBioDataLab/CROssBARv2-KG (AGPL-3.0).
- ◐ Tian ve ark. *DDInter 2.0.* NAR 2025 (lisans: üçüncü taraflara göre CC BY-NC-SA).

Tüm ayrıntılar, düzeltmeler ve eksik bulunan kaynaklar ajan çıktılarında mevcut. Makale yazımında her kaynak yayıncı sayfasından yeniden kontrol edilmeli.

---

## Ek B. Bu klasördeki dosyalar

| Yol | İçerik |
|---|---|
| `analysis/RAPOR_2026.md` | Bu rapor |
| `analysis/reeval/` | Ampirik yeniden değerlendirme betikleri (`s01`–`s06`), `RESULTS.md` (İngilizce, tüm tablolar), `out/*.json` |
| `analysis/reeval_verify/` | Bağımsız sıfırdan doğrulama (`v00`–`v05`) |
| `analysis/pilot_scale/` | 3.618 ilaçlık havuzda düzgün ve hub örneklem pilotu (`p1_*.py`, `RESULTS.md`, `out/`) |
| `analysis/ddiben_pilots/` | DDI-Ben üzerinde ana-etki / derece tabanları, CROssBAR hedef-kNN pilotu ve üç bağımsız doğrulama (DDI-Bench klonu gerekir; bkz. klasördeki README) |

Çalıştırma: repo kökünden `python analysis/reeval/s01_characterize.py` vb. Gerekenler: numpy, scipy, scikit-learn, pandas, networkx; bazı adımlar için rdkit ve torch (CPU). Betikler yalnızca `data/` klasörünü okur. DrugBank'ten türetilmiş ham çift listeleri bu klasöre kopyalanmadı.
