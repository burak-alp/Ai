# Cognitive Core + Externalized Long-Term Memory
## Eleştirel Değerlendirme ve Teknik Araştırma Planı

> Bu doküman bir araştırma önerisinin **hakem gözüyle** eleştirisidir. Amaç fikri onaylamak değil,
> hangi kısımlarının ayakta kaldığını, hangilerinin literatürde zaten çözülmüş olduğunu ve
> hangilerinin matematiksel olarak yanlış kurgulandığını ayırmaktır.
>
> Tarih: 2026-09-11 · Durum: pre-proposal review

---

# 0. Yönetici Özeti — Doğrudan Hüküm

## 0.1 Kısa cevap

**Fikir teorik olarak savunulabilir, ama sunulan haliyle üç yerde kırılıyor ve merkez iddiası
(“500 GB–2 TB monolitik model yerine 5–15 GB core”) bugünkü kanıtlarla ~10x abartılı.**

Fikrin *yönü* doğru: semi-parametric dil modelleri gerçekten parametre karşılığında retrieval
satın alabiliyor, ve bu 2019'dan beri (kNN-LM, REALM, RETRO) ölçülen bir şey. 2024–2026 arası
literatür bunu daha da ileri götürdü: **Memory³** bilgiyi diske externalize edilmiş explicit
KV memory'ye taşıyıp 2.4B'lik bir modeli sıfırdan eğitti; **Memory Layers at Scale** 1.3B'lik bir
modeli factual QA'de Llama-2-7B seviyesine çıkardı. Yani hipotezin zayıf formu **zaten doğrulanmış**.

Sorun şu: senin önerdiğin şeyin %70'i bu iki makalede *zaten var*. Geri kalan %30 ise ya
matematiksel olarak boş (context-dependent meaning), ya da henüz kimsenin yapmadığı asıl değerli
kısım (anti-memorization objective, derived-fact write-back, ölçüm çerçevesi). Projenin hayatta
kalması bu %30'a odaklanmasına bağlı.

## 0.2 Gördüğüm üç temel hata

### Hata 1 — Parametre aritmetiği tutmuyor

Allen-Zhu & Li'nin knowledge capacity scaling law'u: bir transformer parametre başına **~2 bit**
bilgi depolayabiliyor (int8'e quantize edilse bile). Buradan:

```
7B model  → 14 Gbit  ≈ 1.75 GB  saf factual knowledge kapasitesi
```

Yani bir frontier modelin 500 GB'lık ağırlıklarının **ansiklopedik bilgiye giden kısmı birkaç GB
mertebesinde**. Geri kalan parametreler *hesap yapıyor* — FFN katmanları aynı anda hem key-value
memory hem de nonlineer feature dönüşümü. Bilginin %100'ünü mükemmel şekilde dışarı çıkarsan bile
500 GB → 5 GB elde edemezsin, çünkü çıkardığın şey zaten 500 GB'ın küçük bir dilimiydi.

Gerçekçi tavan, literatürün ölçtüğü yer: **factual görevlerde 3–5x aktif parametre tasarrufu**
(Memory Layers: 1.3B + memory ≈ 7B dense). Doğru karşılaştırma da zaten monolitik dense değil,
MoE: DeepSeek-V3 sınıfı bir model 671B total / 37B active. Senin rakibin 500 GB değil, ~74 GB
aktif. 74 GB → 10 GB bile ~7x demek ve bu iyimser uç.

> **Düzeltme:** Hedefi “50–100x parametre tasarrufu” değil, **“knowledge-per-active-parameter
> Pareto eğrisini 3–5x iyileştirmek + bilgiyi editable yapmak”** olarak yeniden yaz. İkincisi
> aslında daha büyük pratik kazanç ve sen onu az önemsiyorsun.

### Hata 2 — `Meaning = f(M, Context)` fikri, bu haliyle ya zaten doğru ya da bilgi-teorik olarak boş

Transformer'ın residual stream'i **zaten** context-dependent bir anlam hesaplıyor. Bir latent'in
farklı context'lerde farklı işlev görmesi yeni bir şey değil; superposition literatürünün
(Elhage et al., *Toy Models of Superposition*) tam konusu bu. Johnson–Lindenstrauss gereği
d boyutta exp(O(ε²d)) tane yaklaşık-ortogonal vektör paketleyebilirsin — ağlar bunu kendiliğinden
yapıyor.

Fikri **yeni** kılmak için nereden ekstra kapasite geldiğini söylemen lazım. Formal cevap:
memory item `M`'i context `C` koşuluyla kodlarsan, ulaşılabilir en iyi hız `H(M|C)`'dir.
Yani kazancın **tam olarak** şudur:

```
Kazanç = H(M) − H(M|C) = I(M ; C)
```

Bu Slepian–Wolf / conditional source coding sınırı. Sonuçları acımasız:

| Bilgi türü | I(M;C) | Sıkıştırma kazancı |
|---|---|---|
| Türetilebilir (F=ma, birim dönüşümü, kimyasal stokiyometri) | yüksek | **büyük** — hatta hiç saklamana gerek yok |
| Yapısal/şablonlu (X ülkesinin başkenti Y) | orta | orta |
| Keyfi long-tail (Planck sabiti, bir datasheet'teki pin numarası, doğum tarihi) | **≈ 0** | **≈ 0** |

**Kritik sonuç:** Senin 2. fikrin (context-dependent multi-meaning) ile 3. fikrin (ezberleme,
türet) **aynı fikrin iki yüzü**. İkisi de I(M;C)'yi sömürüyor. Ve bu, en çok dışarı çıkarmak
istediğin şeyde — datasheet'ler, ürün bilgileri, coğrafya, uzun kuyruk — **çalışmıyor**, çünkü
o bilgiler tanım gereği algoritmik olarak sıkıştırılamaz.

Yani türetme fikri ile long-tail externalization fikri **dağılımın farklı bölgelerine hitap
ediyor ve birbirinin yerine geçmiyor**. Öneride bu ikisi tek bir mekanizmaymış gibi sunulmuş.

### Hata 3 — “Pretrained core'u dondur, memory ekle” planı ana hipotezi test *edemez*

OLMo tabanlı bir core zaten factual knowledge'ı ağırlıklarına ezberlemiş durumda. Üstüne memory
ekleyip “bak kullanıyor” diyebilirsin ama **parametre tasarrufu iddiasını asla kanıtlayamazsın**,
çünkü bilgi hâlâ içeride. Dahası, ölçülmüş bir engel var: *Quantifying Prior Dominance in RAG
Systems* çalışması, açık talimata rağmen ticari bir modelin çelişkili durumların **%47.1'inde**
retrieved context'i yok sayıp parametrik prior'una döndüğünü raporluyor.

> **Düzeltme:** İki ayrı iddia, iki ayrı deney.
> - **Mekanizma iddiası** (“memory reasoning loop'una entegre olabiliyor”) → 1B adaptasyon ile test edilir. ✅ planın uygun.
> - **Tasarruf iddiası** (“daha az parametre ile aynı bilgi”) → **sadece sıfırdan, iso-compute, kontrollü korpus ile** test edilir. 50M–400M ölçeğinde. ❌ planın bunu içermiyor.

## 0.3 Fikrin gerçekten sağlam olan tarafları

Eleştirinin adil olması için: üç noktada haklısın ve literatür seni destekliyor.

1. **Reasoning ve factual knowledge veri düzeyinde ayrışıyor.** Ruis et al. (influence functions
   ile) gösterdi ki factual sorularda her soru *farklı* dokümanlardan besleniyor, ama reasoning
   sorularında aynı dokümanlar *aynı task içindeki farklı sorulara* benzer etki yapıyor —
   yani procedural knowledge diye ayrı ve genellenebilir bir şey var. Bu, ayrıştırma hipotezinin
   en güçlü ampirik dayanağı.

2. **Ezberlemeyi engellemek için doğru kaldıraç veri dağılımı.** Chan et al. gösterdi ki
   burstiness, çok sayıda nadir sınıf ve **item anlamlarının dinamik olması** in-context learning'i
   tetikliyor; in-weights learning ile *takas ilişkisinde*. Senin “değişken sentetik evrenler”
   fikri bu bulgunun doğrudan operasyonelleştirilmiş hali. Teorik temeli var.

3. **Recurrent core consumer donanımda gerçekten mantıklı** — ama senin verdiğin gerekçeden değil.
   Asıl kazanç bandwidth değil **VRAM kapasitesi**: 8 GB'a “30 katman eşdeğeri” sığdırabiliyorsun.
   Bunu öneride belirtmemişsin, en güçlü argümanlarından biri.

---

# Part I — 30 Soruya Cevaplar

## A. Teorik Temeller (S1–S7)

### S1. Yaklaşım teorik olarak mantıklı mı?

**Zayıf formu: evet, kanıtlanmış. Güçlü formu: hayır, henüz kanıtlanmamış ve muhtemelen yanlış.**

- *Zayıf form* — “Retrieval, parametrik bilginin bir kısmının yerini alabilir.” Bu 2019'dan beri
  ölçülüyor (kNN-LM, RETRO). 2026'da *To Memorize or to Retrieve* bunu 30M–3B ölçeğinde
  sistematik scaling law'a döktü: retrieval, tüm model ölçeklerinde parametric-only baseline'ı
  aşıyor. Tartışma bitti.
- *Güçlü form* — “Reasoning ile knowledge temiz biçimde ayrılabilir ve core bilgiden arındırılınca
  reasoning zarar görmez.” Bu **bir varsayım**, ve karşı kanıt var (bkz. S5).

Ayrıca dikkat: vizyon seviyesinde bu fikir **kamuya açık biçimde dolaşımda**. Karpathy'nin
“cognitive core” formülasyonu (ansiklopedik bilgiyi maksimum ölçüde feda eden birkaç milyar
parametrelik model) senin 1. fikrinle neredeyse birebir aynı. Yani *vizyon* novelty değil;
novelty **mekanizma + ölçüm** olmak zorunda.

### S2. Hangi kısımlar yeni, hangileri yapılmış?

| Senin fikrin | Durum | Önceki çalışma |
|---|---|---|
| Bilgiyi model dışına, diske taşımak | ❌ **Yapılmış** | **Memory³** (2024): explicit memory, diskte KV, sıfırdan 2.4B, iki-aşamalı pretraining, memory sparsification |
| Retrieval'ı prompt yerine reasoning loop'una gömmek | ❌ **Yapılmış** | RETRO (chunked cross-attention, alt katmanlar), Memorizing Transformers (kNN-augmented attention), kNN-LM (üst katman) |
| Seyrek aktive edilen büyük knowledge store | ❌ **Yapılmış** | Memory Layers at Scale (128B memory param), Product-Key Memory, PEER |
| Test-time öğrenen uzun dönem bellek | ❌ **Yapılmış** | **Titans** (surprise-based neural LMM, adaptive forgetting) |
| Recurrent latent reasoning, depth ≠ size | ❌ **Yapılmış** | Universal Transformer, **Huginn** (recurrent depth), Coconut, Hierarchical Reasoning Model |
| Context-koşullu representation üretimi | ❌ **Yapılmış** | Hypernetworks, FiLM, superposition literatürü |
| Retrieval vs memorization scaling law'u | ❌ **Yapılmış (2026)** | *To Memorize or to Retrieve* |
| Modelin memory'yi gerçekten kullanıp kullanmadığını ölçmek | 🟡 **Yeni başlamış** | *Quantifying Prior Dominance* (NCU metriği) |
| **Ezberlemeyi aktif olarak cezalandıran training objective** | ✅ **YAPILMAMIŞ** | — |
| **Türetilmiş fact'i memory'ye geri yazan amortized derivation loop** | ✅ **YAPILMAMIŞ** | Titans yazıyor ama context sıkıştırma için, türetilmiş bilgi memoization'ı için değil |
| **I/O bütçesi loss'ta olan, öğrenilmiş 3-katmanlı memory controller** | ✅ **YAPILMAMIŞ** | Memory³'te sabit şema var, öğrenilmiş değil |
| **“Externalizability spectrum”un ampirik haritası** | ✅ **YAPILMAMIŞ** | — |

Özetle: **mimari novelty'n yok, training ve ölçüm novelty'n var.** Projeyi buna göre kur.

### S3. Ayrışabileceğin gerçek noktalar

Dört tane görüyorum, önem sırasıyla:

1. **Anti-memorization objective.** Yukarıdaki tüm çalışmalar memory *ekleyip* modelin onu
   kullanmasını *umuyor*. Kimse “ezberleme” yi bir loss terimi yapmadı. Bu yapılabilir (bkz. S19).
2. **Capacity starvation ile bilgi-teorik garanti.** Core'u fact set'ini ezberleyemeyecek kadar
   küçük tut: `2 × params < fact_set_bits`. Bu bir umut değil, bir ispat. Kimse deneyi böyle
   kurmadı ve Allen-Zhu'nun yasasının doğrudan, zarif bir uygulaması.
3. **Derive-or-retrieve routing + write-back.** Model `F = m·a`'dan 37.1 N türetiyor ve sonucu
   memory'ye *yazıyor*. Bir sonraki sefer lookup. Bu, memory'yi öğrenilmiş bir derived-fact
   cache'ine çeviriyor ve türetilebilir kapanımı ampirik olarak ölçmeni sağlıyor.
4. **Ölçüm çerçevesi.** “Parametric leakage”, “counterfactual fidelity”, “memory dependence”
   metrikleri + bunları ölçen public benchmark. Negatif sonuçta bile yayınlanabilir olan kısım bu.

### S4. Reasoning ile factual storage pratikte ne kadar ayrıştırılabilir?

**Kısmen — ve ayrım ikili değil, süreklilik.** Üç katman var:

```
[1] Saf prosedür        → türev alma kuralı, birim analizi, tümdengelim şeması, kod semantiği
[2] Şematik/ilişkisel   → "X ülkesinin başkenti Y", "A şirketi B'yi satın aldı" kalıpları
[3] Keyfi atom          → Planck sabiti, bir MCU'nun pin dizilimi, 1087 tarihi
```

[3] temiz biçimde externalize edilebilir. [1] edilemez — o *core'un kendisi*. [2] tartışmalı ve
asıl araştırma alanı orası.

Ama daha derin bir problem var: **dilin kendisi bir lookup table.** Sözlük, morfoloji, deyim,
eşdizimlilik — bunlar “bilgi” mi “dil yeteneği” mi? “Water boils at 100°C” bir fact mı yoksa
“water” kelimesinin anlamının parçası mı? Kelime anlamı **dünya bilgisidir**. Bu yüzden
“%100 bilgi-arındırılmış core” diye bir şey yok; olsa olsa **bir arındırma derecesi** var.

> **Öneri:** Bunu bir kusur değil, **ölçülecek bir eğri** olarak ele al. “Externalizability
> spectrum”: her bilgi sınıfı için, externalize edildiğinde performansın ne kadar düştüğü.
> Bu haritanın kendisi yayınlanabilir bir katkı.

### S5. Factual knowledge çıkarılınca reasoning zarar görür mü?

**Büyük ihtimalle evet, ve bu projenin en büyük teknik riski.** İki bağımsız kanıt zinciri:

1. **Procedural entanglement.** Ruis et al.'ın bulgusu iki yönlü okunur: reasoning ayrı bir veri
   kümesinden besleniyor (senin lehine) ama o veri kümesi **kod ve çalışılmış örnek metinlerden**
   oluşuyor — yani reasoning yeteneği, milyonlarca somut *örnek* üzerinden interpolasyon.
   Örnekleri taşıyan şey ise... bilgi.

2. **Reasoning bir retrieval mekanizması.** Google Research'ün *Thinking to recall* çalışması,
   reasoning trace üretmenin parametrik bilgiyi “açtığını” gösteriyor: computational buffer
   etkisi + **factual priming**. Yani CoT kısmen, ağırlıklardaki bilgiye erişim aracı. Ağırlıklardan
   bilgiyi çıkarırsan, reasoning'in üzerinde çalıştığı zemini de çıkarmış olabilirsin.

Analoji: bir insandan tüm alan bilgisini alıp “saf mantık” bıraktığında elde ettiğin şey iyi
bir muhakemeci değil, **hiçbir şey hakkında fikir yürütemeyen bir mantık motoru**. Uzmanlık
büyük ölçüde örüntü tanımadır ve örüntüler örneklerde yaşar.

**Bunu bir kill-criterion'a çevir:** Faz 1'de core'dan factual yoğunluğu düşürdüğünde GSM8K /
ARC-Challenge / BBH'de düşüş **%15'i aşıyorsa**, hipotezin güçlü formu ölmüştür; zayıf forma
(sadece long-tail externalization) çekil.

### S6. Context-dependent memory representation matematiksel olarak nasıl tasarlanır?

Somut ve savunulabilir bir formülasyon:

**Depolama.** Her memory item `M` bir sparse code olarak saklanır:
```
z ∈ R^k ,  ||z||₀ ≤ s          (s ≈ 8–32)
D₀ ∈ R^(d×k)                    paylaşılan sabit dictionary (codebook)
M̂ = D₀ z                        context'siz rekonstrüksiyon
```

**Context-koşullu decoding.** Dictionary'yi düşük ranklı bir karışımla modüle et:
```
D(C) = D₀ + Σᵢ αᵢ(C) · Uᵢ Vᵢᵀ          i = 1..r,  r ≪ k   (low-rank hypernetwork)
α(C) = softmax(W_α · h_C)               context'ten gelen karışım ağırlıkları
M_eff = D(C) · (z ⊙ g(C))               FiLM-tarzı gating
```

**Okuma.** Core, `M_eff`'e cross-attention ile bakar; recurrent adım içinde:
```
q_t   = W_q h_t
z_{1:k} = Retrieve(q_t)                  ANN, non-differentiable (top-k)
M_eff = Interpret(z_{1:k}, h_t)
h_{t+1} = F(h_t, x, M_eff)
```

**Kritik regularizer — memory collapse'i önleyen şey.** Eğer `D(C)` serbestçe modüle edilirse,
dejenere çözüm şudur: context bütün işi yapar, `z` hiçbir bilgi taşımaz. Model memory'ye bakıyormuş
gibi yapar, aslında parametrik prior'undan cevaplar. Bunu engellemek için `z`'nin **sabit** bir
decoder ile fact'i geri verebilmesini zorunlu kıl:

```
L_mem = L_task + β·‖M − D₀z‖²  +  γ·KL(q(z|M) ‖ p(z))
                  └─ context'siz ─┘   └─ information bottleneck ─┘
                     rekonstrüksiyon
```

`β` terimi olmadan bu mimari **sessizce başarısız olur ve başarılı görünür**. Bu, tasarımın en
kolay gözden kaçan noktası.

**Sınır.** Tüm bu makine `H(M|C)`'yi aşamaz. Elde ettiğin şey kapasite değil, **yapısal ön-bilgi**:
örnek verimliliği ve yorumlanabilirlik kazanırsın, ham kapasite kazanmazsın.

### S7. Paylaşılan latent basis'lerden gerçek compression çıkar mı?

**Semantik anlamda: sınırlı. Mühendislik anlamında: evet, ve hayati.**

- *Semantik kazanç* korelasyonla orantılı, yani `I(M;C)` ile. “Flow” soyutlaması akım/akışkan/ısı/
  bilgi arasında paylaşılabilir — ama bu zaten embedding uzaylarında olan bir şey ve kazancı
  ölçüldüğünde mütevazı.
- *Asıl kazanç burada değil.* Paylaşılan codebook + vector quantization, latent memory'yi
  **depolanabilir kılan tek şey.** Aşağıdaki aritmetik projenin en önemli sayısı (bkz. S8).

> **Yeniden çerçeveleme:** Compositional representation fikrini “bir bit'e daha çok anlam
> sığdırma” felsefesi olarak değil, **latent memory store'un product quantization'ı** olarak sun.
> O zaman ölçülebilir, savunulabilir ve zorunlu bir bileşen olur.

---

## B. Memory Tasarımı (S8–S12)

### S8. Latent memory vs metin memory

**Önce herkesi susturan aritmetik.** 1B core: 16 katman, d_model=2048, GQA ile d_kv=512.

| Depolama biçimi | 512-token chunk | Metne oran | 100 GB metin → |
|---|---|---|---|
| Ham metin | ~2 KB | 1x | 100 GB |
| **Tüm katman KV** (fp16) | **16 MB** | **~8000x** | **800 TB** ❌ |
| Tek katman KV (fp16) | 1 MB | ~500x | 50 TB ❌ |
| Chunk-summary, 32 vektör × 2048 (fp16) | 131 KB | ~65x | 6.5 TB ❌ |
| **Chunk-summary + PQ (64 B/vektör)** | **2 KB** | **~1x** | **~100–200 GB** ✅ |

**Sonuç: latent memory metinden daha kompakt değildir — çok daha büyüktür.** Memory³'ün
“memory sparsification” mekanizmasını icat etmek zorunda kalmasının sebebi tam olarak bu.
Tek yaşayabilir tasarım: **token başına KV değil, chunk başına az sayıda summary vektör,
product-quantized.** Bunu baştan kabul et, yoksa 3. ayda SSD'n dolar.

**Latent'in avantajları:** read-time encode maliyeti yok; yüzey formu değil *işlenmiş* içerik
saklanabilir; arayüz differentiable; metinde olmayan şeyler (belirsizlik, yapı) taşınabilir.

**Latent'in problemleri:** (i) yukarıdaki depolama patlaması; (ii) drift (S9); (iii)
**insan tarafından okunamaz** — provenance, citation, denetlenebilirlik, hata ayıklama gider ki
bunlar RAG'ın pratikteki en büyük erdemleri; (iv) modeller arası taşınamaz; (v) chicken-and-egg:
memory'yi kurmak için core lazım, core'u eğitmek için memory lazım.

> **Tavsiye: hibrit.** Ground truth = metin/bayt. Latent = versiyon etiketli, PQ'lu **cache**.
> Bu aynı anda S9'u da çözer.

### S9. Latent memory'nin drift ile bozulması

Gerçek ve ciddi. RETRO bunu retriever'ı **dondurarak** çözdü — bedeli: retriever LM ile birlikte
adapte olamadı. REALM asenkron index yenileme kullandı ve staleness problemini literatüre soktu.
Memorizing Transformers'ın somut azaltıcısı **key/query normalization** idi.

Öneri, ucuzdan pahalıya (1+2+4'ü birlikte uygula, ablate et):

1. **Versiyon etiketi + lazy re-encode.** Her latent'e `core_version` yaz. Erişimde versiyon
   farkı eşiği aşarsa metinden yeniden encode et ve cache'i tazele. Sıcak öğeler doğal olarak
   taze kalır; maliyet amortize olur. *(Hibrit depolamanın bedava gelen faydası.)*
2. **Read-adapter re-alignment.** Memory'yi dondur, **okuyucuyu** adapte et: küçük bir
   `A_t : M_space → core_space_t` adaptörü her checkpoint'te ucuza yeniden eğitilir. TB'larca
   veriyi yeniden encode etmek yerine birkaç milyon parametre eğitirsin.
3. **Anchor loss.** Memory encoder'ın sabit bir probe set üzerindeki çıktısının referans
   basis'ten sapmasını cezalandır; ya da EMA ile güncellenen dondurulmuş codebook kullan.
4. **Key/query normalization.** Ucuz, kanıtlanmış.
5. **Major version'da tam re-encode.** Pahalı ama basit; sadece büyük sürüm sınırlarında.

> **Bu arada bu bir yayın fırsatı:** “core checkpoint mesafesine karşı retrieval recall ve task
> accuracy” eğrisini kimse yayınlamadı. Ölç ve yayınla — küçük ama temiz bir katkı.

### S10. SSD latency/bandwidth ne kadar ciddi?

**Inference'ta ciddi değil. Endişen yanlış yere konumlanmış.** Sayılar:

```
NVMe Gen4 : ~7 GB/s sıralı, ~100 µs rastgele okuma (QD1), ~1M IOPS (yüksek QD)
DDR5 dual : ~60–90 GB/s
RTX 4060  : ~272 GB/s VRAM, PCIe 4.0 x8 ≈ 16 GB/s
```

Retrieval başına 32 vektör × 64 B (PQ) = **2 KB**. Bandwidth açısından gürültü seviyesi.
Inference'ta chunk düzeyinde (64 token'da bir) retrieval, 20 tok/s'de saniyede ~0.3 sorgu.
Reasoning-step düzeyinde, 10 adım/token'da bile ~200 sorgu/s × DiskANN'ın ~4–8 rastgele
okuması ≈ **1.6K IOPS**. NVMe için önemsiz.

**Asıl darboğazlar (doğru endişe listesi):**
1. **Latent depolama boyutu** (S8) — sistemi öldüren şey bu.
2. **Training sırasında ANN CPU throughput'u.** batch 256 × 2048 token / 64 = adım başına 8192
   sorgu. Disk değil, CPU-side arama boğuluyor.
3. **Index staleness** (S9).
4. **Tail latency.** p99 disk okuması p50'nin 10x'i olabilir; recurrent loop içinde bu birikir.

**Mitigasyon:** training memory'sini RAM'de tut; PoC ölçeğinde zaten sığıyor —
100M vektör × 64 B PQ = **6.4 GB**, 32 GB RAM'e rahat oturur. SSD'ye ancak ~1B vektörden
sonra gerçekten ihtiyacın var.

### S11. Retrieval hangi granülaritede?

Yanlış soru — “ya/ya da” değil, **hiyerarşi**. Retrieval granülaritesini bellek hiyerarşisine eşle:

| Katman | Granülarite | Boyut | Ne zaman | Maliyet |
|---|---|---|---|---|
| **SSD → RAM** | Task/query düzeyi, prefetch | 10³–10⁴ öğe | Sorgu başına 1 kez | ~ms |
| **RAM → VRAM** | Reasoning-step düzeyi | k = 8–64 | Adım başına ≤1 | ~100 µs |
| **VRAM içi** | **Token düzeyi *attention*** | resident working set | Her token | bedava |

Ana ilke: **token düzeyinde I/O yanlış, token düzeyinde resident memory'ye attention doğru.**
RETRO'nun 64-token chunk'ı bu trade-off'un ampirik olarak bulunmuş tatlı noktası; oradan başla.

### S12. Memory controller nasıl tasarlanır?

Bunu bir **budget-constrained conditional computation** problemi olarak kur:

```
g_t = σ(w·h_t + b)                       retrieval kapısı
retrieve if g_t > τ
L = L_task + λ · E[ Σ_t c(a_t) ]         c = katman başına I/O maliyeti
```

**Kapıyı eğitmenin pratik hilesi:** RL'e girme. **Oracle gain etiketleri** üret — offline,
teacher-forcing ile her pozisyonda retrieval'lı ve retrieval'sız loss'u hesapla:
```
gain_t = L(h_t | no-retrieval) − L(h_t | retrieval)
```
Sonra kapıyı bu etiketlerle **supervised** eğit. REINFORCE'un varyansından kurtulursun ve
oracle kapı sana tavan performansı (headroom) verir — ablation'da bedava baseline.

Ek zorunlu bileşenler:
- **Refractory period / histerezis:** aynı adımda tekrar tekrar retrieval yapmayı engelle.
- **Retrieval cache:** son N sorguyu ve sonucunu tut; tekrar sorguyu bastır.
- **Hard budget cap:** sorgu başına B retrieval üst sınırı → worst-case latency garantisi.
- **Speculative prefetch:** mevcut plan/adımdan sonraki adımın ihtiyacını tahmin et, SSD→RAM
  transferini reasoning ile örtüştür. (Latency'yi gizlemenin en etkili yolu.)

---

## C. Recurrence (S13)

### S13. Recurrent core ile model boyutu ve reasoning depth gerçekten ayrışır mı?

**Kısmen — ve literatürde ölçülmüş bir tavanı var.**

Huginn (recurrent depth ile test-time compute ölçekleme) çalışıyor: CoT'un aksine özel veri
gerektirmiyor, küçük context ile çalışıyor, kelimeye dökülemeyen reasoning tiplerini
yakalayabiliyor. **Ama** sonraki çalışmalar şunu gösterdi: performans belirli bir iterasyon
derinliğinde tepe yapıyor ve **ondan sonra keskin biçimde düşüyor**. Yani “kolay problem 3,
zor problem 30 iterasyon” tablosu olduğu gibi gerçekleşmiyor; pratikte dar bir kullanışlı
aralık var ve stabilizasyon aktif bir araştırma konusu.

Diğer uyarılar:
- **Recurrence knowledge kapasitesini geri getirmez.** Depth ucuzlar, *bilgi* ucuzlamaz. Bu iki
  fikir (recurrence + externalization) birbirini tamamlar ama birbirinin yerine geçmez.
- **Tam weight-tying representational çeşitliliği öldürür.** 2–4 farklı blok kullan ve *onları*
  döngüye al; tam tying yapma.
- **Latent recurrence'ın gizli kaybı: discretization.** Token-space CoT'ta her adım ayrık bir
  sembole çöküyor — bu bir *hata düzeltme* mekanizması. Latent uzayda döngü kurunca hata
  birikir, çöküş yok. Huginn'in derinlikte bozulmasının muhtemel sebeplerinden biri bu.
- **Consumer donanımda asıl kazanç VRAM kapasitesi**, bandwidth değil. batch-1 decode'da
  her iterasyonda ağırlıkları yeniden okursun (4060'ın 24 MB L2'sine 1B'lik blok sığmaz),
  dolayısıyla bandwidth avantajı yok. Ama 8 GB'a “30 katman eşdeğeri” sığdırman gerçek.

**Tavsiye:** Eğitimde **rastgele unroll derinliği** kullan (Huginn'in yaklaşımı) ki test-time'da
derinlik ayarlanabilsin; truncated BPTT ile geriye yayılımı sınırla; halting head'i zorluk ile
denetimli eğit. Ve adaptive halting'i **ikinci** fazda dene — birinci fazda sabit derinlik ile
çalış, yoksa iki tane kararsız sistemi aynı anda debug edersin.

---

## D. Deney Tasarımı (S14–S19)

### S14. 1B–3B üzerinde hipotezi test etmek için en doğru deney tasarımı

**Ana mesaj: 1B adaptasyonu ana hipotezini test edemez (Hata 3). İki koldan git.**

**Kol A — Mekanizma spike'ı (1B, ~2 hafta).**
OLMo-2-1B base + memory modülleri, core dondurulmuş, sadece yeni modüller eğitiliyor,
sentetik counterfactual evrenler üzerinde. Soru: *memory reasoning loop'una bağlanabiliyor mu?*
Birincil metrik: **counterfactual swap fidelity** — memory'deki değeri değiştir, cevap değişiyor mu?
Bu bir *plumbing* testi, bilimsel iddia değil. Riski erken düşürür.

**Kol B — Bilimsel deney (sıfırdan, 50M–400M, asıl iş).**
Kontrollü korpus üzerinde **eşleştirilmiş** eğitim:
```
Baseline : dense, N parametre
Treatment: N_core parametre + memory store (M öğe)
```
Aşağıdaki eksenleri süpür ve **eğri eğimlerini** karşılaştır:
- Core ölçeği: 50M / 150M / 400M
- Memory ölçeği: 0 / 1x / 5x / 20x fact set
- Token bütçesi: sabit (iso-compute) ve sabit-veri olarak iki kez

Çıktı: *knowledge accuracy vs active parameters* eğrisi. İddia ancak treatment eğrisi baseline
eğrisinden **farklı eğimde** ise kanıtlanır. Sadece yukarıda olması yetmez — o sadece “daha çok
toplam parametre koydum” demek olabilir.

> Not: *To Memorize or to Retrieve* (2026) bu deneyi OLMo-2 tabanlı 30M–3B'de, DCLM ile,
> 3-boyutlu scaling çerçevesinde zaten kurdu. **Sıfırdan kurma — onların protokolünü referans al
> ve senin ayrıştırıcı katkını (anti-memorization loss, capacity starvation) *ek koşul* olarak
> üstüne bindir.** Bu hem zaman kazandırır hem de karşılaştırılabilirlik verir.

**Kontrollü korpusun tasarımı (kritik):**
- Entity'ler **rastgele ID'ler** (yüzey formu prior'u olmasın: “Zeta-7719”, “Qorval” değil “Mars”).
- Her entity'nin attribute'ları; attribute değerleri **evrenden evrene yeniden örneklenir**.
- Aynı attribute'ları *kullanan* prosedürel metin (türetmeler, hesaplar, kod) — prosedürler sabit.
- Evren tutma (held-out universes) ile genelleme ölçülür.
- %40–60 oranında gerçek, dekontamine metin karıştır ki dil yeteneği ölmesin.

### S15. Hangi benchmark'lar?

| Kategori | Set | Neden |
|---|---|---|
| Long-tail bilgi | **PopQA**, EntityQuestions | Tail entity'ler için tasarlandı — externalization'ın vaadinin tam merkezi |
| Açık-alan QA | Natural Questions (open), TriviaQA | Standart karşılaştırılabilirlik |
| Multi-hop | HotpotQA, **MuSiQue** | Tekrarlayan retrieval'ı test eder |
| Bilgiden bağımsız reasoning | GSM8K, MATH, **ARC-Challenge**, BBH | Core'un zarar görüp görmediği (S5 kill-criterion) |
| Sentetik mantık | ProntoQA / ProofWriter | Derinlik ↔ iterasyon ilişkisi temiz ölçülür |
| Güncellenebilirlik | Temporal QA, FreshQA tarzı | Externalization'ın **az iddia ettiğin** en güçlü avantajı |
| Bilgi çelişkisi | Kendi counterfactual suite'in + NCU | Modelin memory'yi mi prior'unu mu kullandığı |
| LM kalitesi | Held-out perplexity, **knowledge-dense vs procedure-dense dilimlere ayrılmış** | Bu ayrım tek başına güzel bir katkı |

**MMLU'yu birincil metrik yapma.** Kontamine ve bilgi ile reasoning'i birbirine karıştırıyor —
senin tam olarak ayırmak istediğin iki şeyi. İkincil olarak raporla.

### S16. “Daha az parametre ile aynı bilgi” iddiasını ölçen metrikler

**Birincil:**
- **Parameter-equivalence ratio (PER):** treatment'ın performansına ulaşmak için dense
  baseline'ın kaç parametreye ihtiyacı var? Tek sayılık, dürüst, karşılaştırılabilir.
- **Bits-per-active-parameter:** sentetik korpusta depolanan bilgi bitlerini Allen-Zhu
  yöntemiyle tahmin et, aktif parametreye böl.

**Externalization'ın gerçekliğini ölçenler (bunlar senin özgün katkın):**
- **Parametric leakage** `L = acc(memory ablated) − acc(random)` → **düşük olmalı**. Yüksekse
  model ezberlemiş demektir.
- **Memory dependence** `D = acc(correct memory) − acc(ablated memory)` → **yüksek olmalı**.
- **Counterfactual fidelity** `CF = P(cevap, memory'deki yeni değere döner | fact değiştirildi)`
  → **asıl metrik bu.** 1.0'a yakın olmalı.
- **NCU** (Normalized Context Utilization, prior-dominance literatüründen) — log-prob tabanlı,
  zero-shot / oracle / adversarial koşulları karşılaştırır. Hazır metriği yeniden icat etme.
- **Derivation gain** `= acc(memory'de yok ama türetilebilir) − acc(memory'de yok ve türetilemez)`
  → senin 3. fikrini doğrudan ölçer.

**Maliyet tarafı (bunları raporlamazsan hakem reddeder):**
aktif parametre · toplam bayt (VRAM+RAM+SSD) · FLOPs/token · tok/s · retrieval/token ·
IOPS/token · J/token.

**En önemli metodolojik kural — 4 yönlü iso-karşılaştırma tablosu:**

| Karşılaştırma | Ne sabit | Neyi dürüstçe test eder |
|---|---|---|
| iso-active-param | aktif parametre | “aynı hesapla daha çok bilgi” |
| iso-FLOP | token başına FLOP | compute avantajı |
| iso-total-bytes | VRAM+RAM+SSD toplamı | **gerçek sıkıştırma** — çoğu makale bunu atlar |
| iso-latency | tok/s | pratik kullanılabilirlik |

Literatürdeki memory/retrieval makalelerinin çoğu bunlardan **birini** seçip diğerlerini
görmezden geliyor. Dördünü birden raporlamak tek başına metodolojik bir katkı.

### S17. Training dataset nasıl hazırlanmalı?

Dört bileşenli karışım:

```
%40–50  Doğal metin (dekontamine Dolma/DCLM dilimi)     → dil yeteneği korunur
%25–35  Sentetik counterfactual evrenler                → memory bağımlılığı öğrenilir
%10–15  Çelişki örnekleri (memory ≠ prior)              → güven kalibrasyonu
%5–10   Eksiklik örnekleri (fact memory'de YOK)         → zarif türetme / "bilmiyorum"
```

Son iki bileşen sıklıkla atlanıyor ve atlanınca sistem dağıtımda çöküyor:
- **Çelişki örnekleri olmadan** model %47 oranında prior'una döner (prior dominance).
- **Eksiklik örnekleri olmadan** model memory miss'te dense baseline'dan **daha çok**
  halüsinasyon yapar — çünkü retrieval'a güvenmeyi öğrenmiş ama boş dönüşü işlemeyi öğrenmemiştir.

Chan et al.'ın tarifine göre veriyi şekillendir: **burstiness** (fact'ler kümeler halinde geçsin),
**çok sayıda nadir sınıf**, **dinamik item anlamları**, ve **Zipfian çarpıklık** — sonuncusu
kritik, çünkü Chan et al. ICL ile in-weights learning'in başta takas ilişkisinde olduğunu ama
Zipfian dağılımda **bir arada var olabildiklerini** buldu. Sen ikisinin bir arada olmasını
istiyorsun (dil için in-weights, fact'ler için in-context), yani Zipfian şart.

### S18. Sentetik değişken evrenler mantıklı mı?

**Evet — ve teorik gerekçesi var** (Chan et al.: dinamik item anlamı → ICL). Ama üç bilinen
tuzağı var:

1. **Shortcut:** model “her şeyi memory'ye yönlendir” kısayolunu öğrenir ve parametrik prior'un
   *faydalı* olduğu yerlerde de onu kullanmayı bırakır. → Karışıma prior'un doğru olduğu
   örnekleri koy.
2. **Transfer boşluğu:** sentetik yüzey formu fazla düzenli; doğal metne geçmiyor.
   → Sentetik fact'leri gerçek metin akışının *içine* göm, ayrı bir bölüm olarak verme.
3. **Ölçü kaçırma:** çok az evren varyasyonu → ezber devam eder. Bunu ablate et:
   1 / 10 / 100 / 1000 evren; ezberlemenin hangi varyasyon sayısında kırıldığını **ölç**.
   Bu eğri kendi başına ilginç bir sonuç.

### S19. Modelin yine ağırlıklara ezberlemesini nasıl engellersin?

Dört mekanizma, **güvenilirlik sırasıyla**:

**1. Capacity starvation (en güçlü — bir umut değil, bir ispat).**
Allen-Zhu'nun yasasını tersine çevir: core'u fact set'ini fiziksel olarak alamayacak kadar küçük tut.
```
2 bit × params  <  fact_set_bits / 10
```
Örnek: 150M parametre → 300 Mbit tavan. Fact set 100M fact × ~50 bit = 5 Gbit.
Ezberleme **bilgi-teorik olarak imkânsız**. Deneyi böyle kur — bu tek başına en temiz
metodolojik hamle ve kimse bunu bu şekilde kullanmadı.

**2. Anti-memorization loss (en özgün).**
Fact `f`, doğru değer `v`. Memory ablate edilmiş koşulda modelin `v` üzerindeki güvenini cezalandır:
```
L = L_task(memory ✓)  +  λ · [ −H( p(answer | x, memory=∅) ) ]
```
Yani memory yokken model **belirsiz olmalı**. Denk bir formülasyon: `I(output ; v | memory)`'yi
minimize et. Pratikte: her batch'te bir alt-kümeyi memory'siz forward'la ve entropiyi ödüllendir.
λ'yı süpür; çok yüksekte model memory varken de belirsizleşir (kolayca tespit edilir).

**3. Value resampling per epoch.**
Attribute değerlerini her epoch'ta yeniden örnekle. Ezberlemeye giden gradyan **aktif olarak
yıkıcı** hale gelir — dün öğrendiği değer bugün yanlış. Ucuz ve çok etkili.

**4. Entity aliasing.**
Gerçek isim kullanma; rastgele ID ver. Yüzey formu prior'unu ve kontaminasyonu aynı anda öldürür.

**Doğrulama:** Bunların işe yaradığını **parametric leakage** metriği ile ölç (S16). Ölçmezsen
bilemezsin, ve prior dominance literatürü gösteriyor ki varsayım yanlış çıkıyor.

---

## E. Uygulama (S20–S24)

### S20. Hangi base model?

Önerinde bir düzeltme: **OLMo 3'ün 1B'si yok.** OLMo 3 ailesi 7B ve 32B olarak çıktı
(Base / Think / Instruct / RL-Zero), Dolma 3 (~9.3T token) üzerinde. 1B isteniyorsa **OLMo 2 1B**
doğru seçim.

| Amaç | Model | Gerekçe |
|---|---|---|
| **Kol A mekanizma spike'ı** | **OLMo 2 1B Base** | Tam açık (veri + checkpoint + kod). *Verinin açık olması burada opsiyonel değil* — ağırlıklarda hangi fact'lerin olduğunu bilmeden leakage ölçemezsin. |
| **Kol B sıfırdan scaling** | Kendi 50M/150M/400M'in, OLMo/DCLM mimarisi ile | Sıfırdan eğitim şart; hazır model kullanılamaz |
| Faz 3 ölçekleme | **OLMo 3 7B Base** | Aynı tam-açıklık, güçlü baseline, Dolma 3 dekontaminasyonu |
| Yedek/karşılaştırma | SmolLM2/3 (açık veri), Pythia (küçük scaling ladder) | Pythia'nın 70M–410M merdiveni Kol B için kullanışlı |

Qwen/Llama sınıfı modelleri **birincil yapma** — veri açık değil, leakage ölçümün geçersiz olur.

### S21. Hangi modüller eklenmeli?

```
1.  Memory Encoder         chunk → s adet summary latent  (+ PQ quantizer)
2.  Memory Index           katmanlı ANN: RAM'de kaba seviye, SSD'de ince (DiskANN/SPANN tarzı)
3.  Query Head             h_t → q_t ; ayrıca hangi katmandan okunacağını seçer
4.  Retrieval Gate         ne zaman retrieval yapılacağı, bütçe-farkında  (S12)
5.  Memory Cross-Attention derinliğin ~1/3 ve ~2/3'ünde + üst katmanda;
                           ** zero-init gating ** ile — pretrained core bozulmasın
6.  Context Interpreter    D(C) low-rank hypernetwork + FiLM gating  (S6)
7.  Recurrent Block        2–4 farklı blok, döngülenmiş + Halting Head
8.  Write-Back / Derivation Cache  + confidence estimator (sadece yüksek güvenli türetmeler yazılır)
9.  Read-Adapter           drift telafisi, versiyonlu  (S9)
10. Conflict / Trust Head  memory ile prior çeliştiğinde hangisine güvenileceği
```

**Pratik uyarı (6 numara):** Context Interpreter'ı **Faz 1'de ekleme.** En kırılgan, en az
kanıtlanmış ve sessizce dejenere olan bileşen o. Önce 1–5 ile çalışan bir sistem kur, sonra
6'yı bir ablation olarak ekle. Aksi halde beş yeni başarısızlık modunu aynı anda debug edersin.

### S22. Mimari diyagram

```
                                    ┌─────────────────────────────────────┐
                                    │  INPUT  x                           │
                                    └───────────────┬─────────────────────┘
                                                    ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  VRAM  (4–8 GB)                                                              │
  │                                                                              │
  │   ┌──────────────────────── COGNITIVE CORE ──────────────────────────────┐  │
  │   │                                                                       │  │
  │   │   h_t ──► [ Recurrent Block ×(2–4, looped) ] ──► h_{t+1}             │  │
  │   │             ▲                    │                                    │  │
  │   │             │                    ▼                                    │  │
  │   │             │            [ Halting Head ] ──► dur / devam            │  │
  │   │             │                                                         │  │
  │   │   ┌─────────┴──────────┐                                             │  │
  │   │   │ Memory Cross-Attn  │ ◄── M_eff   (zero-init gated)               │  │
  │   │   │  @ L/3, 2L/3, top  │                                             │  │
  │   │   └─────────▲──────────┘                                             │  │
  │   └─────────────┼─────────────────────────────────────────────────────────┘  │
  │                 │                                                            │
  │        ┌────────┴─────────┐      ┌──────────────────┐    ┌───────────────┐  │
  │        │ Context Interp.  │◄─────│  Read-Adapter A_t│◄───│ Working Set   │  │
  │        │  M_eff=D(C)(z⊙g) │      │  (drift telafi)  │    │ k=8..64 latent│  │
  │        └──────────────────┘      └──────────────────┘    └───────▲───────┘  │
  │                                                                   │          │
  │   ┌───────────────┐   ┌──────────────┐   ┌─────────────────┐     │          │
  │   │  Query Head   │──►│Retrieval Gate│──►│ Conflict/Trust  │     │          │
  │   │   q_t=W_q h_t │   │ g_t>τ, budget│   │      Head       │     │          │
  │   └───────────────┘   └──────┬───────┘   └─────────────────┘     │          │
  └──────────────────────────────┼─────────────────────────────────── │──────────┘
                                 │ step-level retrieval               │
                                 ▼                                    │
  ┌──────────────────────────────────────────────────────────────────┴──────────┐
  │  RAM  (16–32 GB)        Recent memory · Coarse ANN index · Retrieval cache   │
  │                         ~100M PQ vektör ≈ 6.4 GB                             │
  └──────────────────────────────┬───────────────────────────────────────────────┘
                                 │ task-level prefetch (speculative)
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  SSD  (100 GB – 2 TB)                                                        │
  │    ┌────────────────────────┐   ┌──────────────────────────────────────────┐ │
  │    │ GROUND TRUTH: metin/bayt│   │ LATENT CACHE: PQ chunk-summary          │ │
  │    │ (provenance, re-encode) │◄──│ + core_version etiketi (lazy refresh)   │ │
  │    └────────────────────────┘   └──────────────────────────────────────────┘ │
  │    ┌──────────────────────────────────────────────────────────────────────┐  │
  │    │ DERIVED-FACT CACHE  ◄── Write-Back (yalnız yüksek güvenli türetmeler) │  │
  │    └──────────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────────────┘
```

**Okuma döngüsü:**
```
h₀ = Embed(x)
for t in 1..T:
    if Gate(h_t) and budget_remaining:
        q_t   = QueryHead(h_t)
        z     = Retrieve(q_t)                  # RAM working set, gerekirse SSD
        M_eff = Interpret(ReadAdapter(z), h_t)
        h_t   = h_t + γ · CrossAttn(h_t, M_eff)   # γ zero-init
    h_{t+1} = RecurrentBlock(h_t, x)
    if Halt(h_{t+1}): break
y = Decode(h_T)
if Confidence(y) > θ and y türetilmiş:
    WriteBack(y → derived-fact cache)
```

### S23. Training aşamaları

| Faz | Ne eğitiliyor | Donuk | Amaç | Çıkış kriteri |
|---|---|---|---|---|
| **0** | — | — | Korpus + memory store inşası, dekontaminasyon, index | Store hazır, leakage taraması temiz |
| **1** | Memory Encoder | Core | Rekonstrüksiyon + contrastive retrieval | Retrieval recall@10 > 0.9 sentetikte |
| **2** | Cross-Attn + Gate (zero-init) | Core, Encoder | Read-path warmup | Counterfactual fidelity > 0.5 |
| **3** | Core (düşük LR) + hepsi | — | **Continued pretraining, memory in-loop + anti-memorization loss + counterfactual evrenler** | Parametric leakage < 0.1; dil PPL bozulması < %3 |
| **4** | Controller | Diğerleri | Bütçe-farkında kapı, oracle-gain supervision | Oracle kapının %90'ı, retrieval sayısı %50 azalmış |
| **5** | Recurrent Block + Halting | — | Rastgele unroll derinliği, truncated BPTT | Derinlik–doğruluk eğrisi monoton (çökmüyor) |
| **6** | Write-back policy | — | Derive-vs-retrieve routing, confidence kalibrasyonu | Derivation gain > 0, yanlış yazma oranı < %1 |
| **B** | *(paralel kol)* Sıfırdan 50M/150M/400M | — | **Scaling iddiası** | Eğim farkı istatistiksel anlamlı |

**Faz sırası önemli:** 2'den önce 3'e geçme (zero-init gating ile core'u koru), ve 5'i 3'ten
sonra yap — recurrence ile memory'yi aynı anda öğretmeye çalışırsan hangisinin bozulduğunu
ayıramazsın.

### S24. Ablation çalışmaları

**Zorunlu (bunlar olmadan hakem reddeder):**

| # | Ablation | Ne kanıtlar |
|---|---|---|
| 1 | Dense baseline (iso-param, iso-FLOP) | Temel karşılaştırma |
| 2 | **Prompt-level RAG (metin context'e)** | **“Bu zaten RAG” itirazına cevap.** En kritik kontrol. |
| 3 | Memory var ama rastgele içerik | Kazanç bilgiden mi, ekstra hesaptan mı? |
| 4 | Memory ablate (∅) | Parametric leakage ölçümü |
| 5 | Oracle retrieval | Headroom üst sınırı |

**Tasarım ablation'ları:**

| # | Eksen | Değerler |
|---|---|---|
| 6 | Retrieval granülaritesi | token / 64-chunk / step / task |
| 7 | Entegrasyon noktası | alt katman (RETRO) / üst (kNN-LM) / çoklu / gated residual |
| 8 | Memory temsili | metin-reencode / per-token KV / chunk-summary / PQ'lu / codebook-composed |
| 9 | **Anti-memorization loss** | off / λ ∈ {0.01, 0.1, 1.0} |
| 10 | **Evren çeşitliliği** | 1 / 10 / 100 / 1000 — ezberin kırıldığı nokta |
| 11 | Capacity starvation oranı | fact_bits / (2·params) ∈ {0.1, 1, 10, 100} |
| 12 | Recurrence derinliği | 1/2/4/8/16, sabit vs adaptive halting |
| 13 | Controller | always / random / learned / oracle |
| 14 | **Write-back** | on/off → derivation gain |
| 15 | **Drift** | core'u N adım güncelle, bayat memory ile ölç; read-adapter var/yok |
| 16 | Encoder | dondurulmuş vs co-trained (REALM instability testi) |
| 17 | Memory ölçeği | 0 / 1x / 5x / 20x |
| 18 | **Çelişki** | memory ≠ prior → NCU, prior dominance oranı |
| 19 | **Eksiklik** | fact memory'de yok → zarif türetme mi halüsinasyon mu |
| 20 | Context Interpreter | off / FiLM / low-rank hypernetwork; β reconstruction terimi var/yok |

**2, 9, 10, 14, 19** senin özgün iddialarını taşıyan ablation'lar. Bütçe daralırsa
diğerlerini kes, bunları kesme.

---

## F. Risk ve Sonuç (S25–S30)

### S25. Başarısızlığa yol açabilecek en önemli 10 teknik neden

| # | Risk | Olasılık | Erken uyarı sinyali |
|---|---|---|---|
| 1 | **Bilgi çıkınca reasoning çöküyor** (procedural entanglement, factual priming) | **Yüksek** | Faz 3'te GSM8K/ARC düşüşü > %15 |
| 2 | **Model yine de ezberliyor** (prior dominance %47) | **Yüksek** | Parametric leakage > 0.3 |
| 3 | **Latent depolama patlaması** — SSD, değiştirdiğin modelden büyük | **Yüksek** | Faz 0'da store boyutu metnin >5x'i |
| 4 | **Drift** — core güncellendikçe memory geçersizleşiyor, re-encode yetişmiyor | Orta-Yüksek | Checkpoint mesafesi ile recall düşüşü |
| 5 | **Retriever–LM co-adaptation instability** (index modeli kovalıyor, REALM problemi) | Orta | Eğitim sırasında recall salınımı |
| 6 | **Kapı öğrenilemiyor** — non-differentiable top-k; controller always-on veya never-on'a çöküyor | Orta | Gate entropisi 0'a gidiyor |
| 7 | **Recurrence doyuyor/ıraksıyor** (Huginn'in tepe-sonra-düşüş davranışı) | Orta-Yüksek | Derinlik–doğruluk eğrisi tepe yapıyor |
| 8 | **Paylaşılan basis kazancı ≈ 0** tam da externalize etmek istediğin tail fact'lerde (I(M;C)≈0) | **Yüksek** | Codebook ablation'ı fark yaratmıyor |
| 9 | **Değerlendirme confound'u** — kazanç externalization'dan değil ekstra toplam parametre/hesaptan | **Yüksek** | iso-total-bytes karşılaştırmasında avantaj kayboluyor |
| 10 | **Latency** — sistem token başına 5–10x yavaş, pratik kazanç siliniyor | Orta | tok/s < dense baseline'ın %30'u |
| 11 | *(bonus)* Sentetik evrenler doğal metne transfer olmuyor | Orta | Sentetikte iyi, PopQA'da yok |
| 12 | *(bonus)* Memory miss'te dense baseline'dan **daha çok** halüsinasyon | Orta | Eksiklik ablation'ında hata oranı artıyor |

**1, 2, 3, 8, 9 projeyi öldürebilecek olanlar.** Hepsi ilk 6 haftada ucuza test edilebilir —
roadmap'i buna göre kurdum.

### S26. Başarılı olursa gerçek avantaj ne?

Sıralamayı **senin önerinden farklı** yapıyorum, çünkü en büyük kazancı en az vurgulamışsın:

1. **Bilginin yeniden eğitim olmadan güncellenebilmesi.** Frontier modellerin en pahalı,
   en çözülmemiş problemi bu. Bir fact değişti → bir satır yaz. ROME/MEMIT tarzı knowledge
   editing'in tüm kırılganlığı ortadan kalkar. **Asıl satış argümanın bu olmalı.**
2. **Unlearning / uyumluluk.** GDPR silme talebi bir DELETE, bir retrain değil. Ticari olarak devasa.
3. **Provenance ve denetlenebilirlik** (hibrit metin+latent tasarımda) — hangi cevabın hangi
   kaynaktan geldiği izlenebilir.
4. **Kişiselleştirme ve gizlilik** — memory yerel, kullanıcıya özel, cihazda kalabilir;
   core paylaşılır.
5. **Bilgiyi ucuz donanımla ölçeklemek** — GPU değil SSD ekliyorsun. Marjinal maliyet ~100x düşük.
6. **Knowledge-per-active-parameter Pareto iyileşmesi** — 3–5x, frontier değil ama gerçek.

**Vermeyeceği şey:** frontier seviyesi reasoning. Rekabetçi matematik, uzun-ufuklu agentic
coding — bunlar bilgi-sınırlı değil, hesap ve eğitim-sinyali sınırlı. Externalization bu boşluğu
kapatmaz. Bunu makalede açıkça yaz; hakemler tam buraya vuracak.

### S27. RTX 4060 8 GB + 32 GB RAM ile ne yapılabilir?

**Yapılabilenler:**

| İş | Fizibilite | Not |
|---|---|---|
| OLMo-2-1B dondurulmuş + memory modülleri eğitimi | ✅ | bf16, 1B ağırlık 2 GB donuk; adaptör + aktivasyon sığar; seq 1024, micro-batch 1–2 + grad accumulation |
| 1B core 4-bit inference + memory | ✅ Rahat | ~0.6 GB ağırlık, geri kalan working set |
| Sıfırdan ≤200M eğitimi | 🟡 Yavaş ama olur | 150M × 10B token = 9e18 FLOP → 4060'ta ~5 gün |
| RAM'de ANN index | ✅ | 100M vektör × 64 B PQ = **6.4 GB** — SSD'ye bile gerek yok |
| Tam ablation sweep (20 konfig) | ❌ | Aylar sürer |
| 7B ölçeklemesi | ❌ | VRAM yetmez |

**Bütçe tavsiyesi:** 4060'ı geliştirme, debug, inference ve küçük ablation'lar için kullan;
**Kol B'nin scaling sweep'ini kirala.** Aritmetik:
```
150M × 10B token = 9e18 FLOP
H100 @ ~%40 MFU ≈ 1.6e14 FLOP/s  →  ~15.6 saat  →  ~$30–60
6 konfigürasyon → ~$200–400 toplam
```
Bu, projenin bilimsel omurgasını birkaç yüz dolara satın alıyor. Kesinlikle yap.

### S28. 8–16 GB VRAM içinde frontier'a yaklaşmak ne kadar gerçekçi?

**Bilgi-yoğun görevlerde: makul. Genel frontier paritesi: gerçekçi değil.**

Kanıta dayalı tahmin:
- Memory Layers at Scale bugün **1.3B + memory ≈ 7B dense** (factual QA) yapabiliyor. Yani ~5x.
- Senin katkıların bunu belki 2–3x daha iyileştirir → 8 GB'da ~2B aktif core ile
  **~20–30B dense eşdeğeri bilgi performansı.** Bu iddia edilebilir ve savunulabilir.
- Ama frontier modellerin avantajı çoktan bilgi olmaktan çıktı. Reasoning, uzun-ufuk planlama,
  tool kullanımı, RL-sonrası davranış — bunların hiçbiri externalization ile gelmiyor.

**Dürüst hedef cümlesi:** *“8 GB VRAM'de, 20–30B dense sınıfı bilgi performansı + güncellenebilir
bilgi tabanı, ~2B aktif parametre ile.”* Bu hedefe ulaşırsan çok iyi bir makale olur.
*“Frontier in 8 GB”* dersen hakem ilk paragrafta reddeder.

### S29. Yayınlanabilir özgün contribution ne olmalı?

**Mimariyi başa koyma — mimari novelty'n yok (S2).** Şu üçlüyü başa koy:

> **C1 — Objective:** *Memory-grounding / anti-memorization loss* + capacity-starvation deney
> protokolü. Externalization'ı bir umut olmaktan çıkarıp eğitim hedefine çeviren ilk çalışma.
>
> **C2 — Measurement:** Parametric leakage, memory dependence, counterfactual fidelity ve
> derivation gain metrikleri + bunları ölçen **public counterfactual-worlds benchmark**.
> Alan bu araca ihtiyaç duyuyor ve senin dışında kimse yapmıyor.
>
> **C3 — Empirical map:** *Externalizability spectrum* — hangi bilgi sınıfı ne kadar kayıpla
> dışarı çıkarılabilir, ve reasoning hangi noktada bozulmaya başlar.

**Neden bu üçlü:** çünkü **negatif sonuçta bile yayınlanabilir**. “Knowledge externalization
%X'te doyuyor ve sınır tam olarak burası, işte ölçüm aracı” — bu COLM/ICLR için sağlam bir
makale. Oysa “yeni mimari önerdim, 3x daha iyi” iddiası çalışmazsa elinde hiçbir şey kalmaz.

Opsiyonel 4. katkı, çalışırsa çok parlak: **C4 — Amortized derivation** (derive-or-retrieve
routing + derived-fact write-back). Bu gerçekten yeni ve “bilgiyi türet, ezberleme” fikrini
ölçülebilir bir mekanizmaya çeviriyor.

### S30. Ben olsam fikri nasıl değiştirirdim?

Yedi somut değişiklik:

1. **Sıralamayı ters çevir.** 1B adaptasyonunu *ana* deney yapma — o sadece 2 haftalık bir
   plumbing spike'ı. Bilimsel sonuç **küçük ölçekli sıfırdan** koldan gelecek. Mevcut planın
   bunun tersini yapıyor ve bu yüzden ana hipotezi hiç test edemeden 3 ay geçebilir.
2. **Capacity starvation'ı birinci sınıf mekanizma yap.** Umut etmek yerine ispatla.
3. **Context-dependent meaning'i manşetten indir.** Onu “latent memory'nin product
   quantization'ı” olarak yeniden çerçevele — o zaman zorunlu, ölçülebilir ve savunulabilir olur.
   `I(M;C)` sınırını makalede açıkça yaz; hakem zaten soracak, sen sor.
4. **Write-back döngüsünü ekle.** En özgün fikrin bu ve sen onu önerinde sadece ima etmişsin.
5. **Hibrit depolama** (metin ground truth + PQ'lu latent cache). Drift ve provenance
   problemlerini aynı anda çözer.
6. **Hedefi yeniden yaz:** “frontier in 8 GB” değil, **“knowledge-per-active-parameter Pareto'su
   + editable knowledge”**. Aynı iş, savunulabilir iddia.
7. **İlk günden ölçüm altyapısını kur.** Parametric leakage ve counterfactual fidelity'yi
   ölçemiyorsan, üç ay sonra çalışan bir sistemin olacak ama **çalışıp çalışmadığını
   bilemeyeceksin.** Prior dominance literatürü bunun ne kadar kolay olduğunu gösteriyor.


---

# Part II — Proposed Architecture

**CC-XM: Cognitive Core with eXternalized Memory**

## II.1 Tasarım ilkeleri

| İlke | Gerekçe |
|---|---|
| **Retrieval granülaritesi = bellek hiyerarşisi** | Token düzeyinde I/O yanlış; token düzeyinde *resident* memory'ye attention doğru (S11) |
| **Memory = metin (ground truth) + PQ'lu latent (cache)** | Drift ve depolama patlamasını aynı anda çözer (S8, S9) |
| **Zero-init gating** | Pretrained core yeni modüller eklenirken bozulmasın |
| **Capacity starvation** | Ezberlemeyi umut etmek yerine bilgi-teorik olarak imkânsız kıl (S19) |
| **Her yeni modül bir ablation** | 10 bileşeni aynı anda debug etme; fazlara yay (S21, S23) |
| **Bütçe loss'un içinde** | Retrieval maliyeti optimize edilen bir şey olmalı, sonradan ölçülen değil (S12) |

## II.2 Bileşenler ve boyutlandırma (1B PoC hedefi)

| Modül | Parametre | Not |
|---|---|---|
| Cognitive Core (OLMo-2-1B tabanı) | ~1.2B | Faz 2'de donuk, Faz 3'te düşük LR |
| Memory Cross-Attention × 3 | ~60M | L/3, 2L/3, top; zero-init gate γ |
| Memory Encoder | ~80M | chunk(512 tok) → s=32 summary latent |
| PQ Quantizer | — | 64 B/vektör, 256 centroid × 8 alt-uzay |
| Query Head | ~4M | h_t → q_t |
| Retrieval Gate / Controller | ~2M | + bütçe durumu girdisi |
| Context Interpreter (Faz 4+) | ~40M | D₀ ∈ R^(2048×4096), r=8 low-rank |
| Read-Adapter | ~8M | versiyonlu, ucuza yeniden eğitilir |
| Halting Head | ~1M | |
| Confidence / Trust Head | ~2M | write-back ve conflict kararı |
| **Toplam aktif ek** | **~200M** | Core üstüne %17 |

Memory store (PoC): 10M chunk × 32 vektör × 64 B ≈ **20 GB** — SSD'de rahat, kaba index RAM'de.

## II.3 Formal tanım

**Yazma (offline / write-back):**
```
c            : metin chunk (ground truth, diskte kalır)
E(c)         → {m₁..m_s} ∈ R^d          Memory Encoder
z_i          = PQ(m_i)                   64 B kod
store(z_i, core_version, ptr→c)
```

**Okuma (inference / training):**
```
h₀ = Embed(x)
budget = B
for t = 1..T_max:
    if Gate(h_t, budget) :
        q_t   = W_q h_t
        Z     = ANN(q_t, k)                       # RAM working set → gerekirse SSD
        Z     = ReadAdapter_{v}(Z)                # drift telafisi
        M_eff = D(h_t) · (Z ⊙ g(h_t))             # Context Interpreter
        h_t   = h_t + γ · CrossAttn(h_t, M_eff)   # γ zero-init
        budget -= cost(tier)
    h_{t+1} = RecurrentBlock(h_t, x)
    if Halt(h_{t+1}) : break
y = Decode(h_T)
if TrustHead(y) > θ and derived(y) : WriteBack(y)
```

**Eğitim hedefi (tam):**
```
L =  L_LM(memory ✓)                                   # ana görev
  +  λ_mem · [ −H( p(a | x, memory = ∅) ) ]           # anti-memorization  (C1)
  +  λ_cost · E[ Σ_t cost(a_t) ]                      # I/O bütçesi         (S12)
  +  β · ‖M − D₀ z‖²                                  # memory collapse guard (S6)
  +  γ_ib · KL( q(z|M) ‖ p(z) )                       # information bottleneck
  +  δ · L_retrieval(contrastive)                     # encoder/retriever
```

`λ_mem` terimi bu mimarinin literatürdeki tüm benzerlerinden ayrıldığı yer.
`β` terimi olmadan sistem sessizce dejenere olur ve başarılı görünür.

---

# Part III — Novelty Assessment

| Bileşen | Novelty | Karar |
|---|---|---|
| Bilgiyi diske externalize etmek | ❌ Yok (Memory³, 2024) | Uygula, iddia etme |
| Reasoning loop'una retrieval gömmek | ❌ Yok (RETRO, Memorizing Transformers) | Uygula, iddia etme |
| Seyrek büyük knowledge store | ❌ Yok (Memory Layers, PEER) | Uygula, iddia etme |
| Recurrent depth ≠ model size | ❌ Yok (UT, Huginn) | Uygula, sınırlarını raporla |
| Context-dependent latent yorumu | ❌ Yok (hypernetwork, FiLM, superposition) | PQ olarak yeniden çerçevele |
| Retrieval vs memorization scaling | ❌ Yok (*To Memorize or to Retrieve*, 2026) | Protokolünü referans al |
| **Anti-memorization objective** | ✅ **Yeni** | **Manşet katkı C1** |
| **Capacity-starvation deney protokolü** | ✅ **Yeni** | **C1'in parçası** |
| **Externalization ölçüm çerçevesi + benchmark** | ✅ **Yeni** | **Manşet katkı C2** |
| **Externalizability spectrum haritası** | ✅ **Yeni** | **Manşet katkı C3** |
| **Derived-fact write-back / amortized derivation** | ✅ **Yeni** | **Opsiyonel C4** |
| **Bütçe-farkında öğrenilmiş 3-katmanlı controller** | 🟡 Yarı-yeni | İkincil katkı |
| **Latent memory drift eğrisi + read-adapter çözümü** | 🟡 Yarı-yeni | İkincil katkı |

**Tek cümlelik konumlandırma:**

> *“Memory³ ve Memory Layers bilginin dışarı çıkarılabileceğini gösterdi. Biz modelin bunu
> yapmaya **zorlanıp zorlanamayacağını**, **ne kadarının** çıkarılabileceğini ve **reasoning'in
> nerede bozulduğunu** ölçüyoruz.”*

---

# Part IV — Related Work

## IV.1 Semi-parametric dil modelleri
- **kNN-LM** (Khandelwal et al., 2019) — üst katmanda olasılık interpolasyonu; en basit form.
- **REALM** (Guu et al., 2020) — retriever'ı LM ile birlikte eğitti; **asenkron index yenileme**
  problemini literatüre soktu (senin S9 sorunun kökeni).
- **RETRO** (Borgeaud et al., 2021, ✓2112.04426) — **chunked cross-attention**, *alt* katmanlarda
  entegrasyon; trilyon-token store. Retriever **dondurulmuş** — ölçekleme sağladı ama LM ile
  birlikte adapte olamadı.
- **Memorizing Transformers** (Wu et al., 2022) — kNN-augmented attention, öğrenilmiş gate;
  staleness'ı **key/query normalization** ile azalttı.
- **Retrieval-Pretrained Transformer** (TACL 2024) — self-retrieval ile uzun menzilli modelleme.
- **On the Generalization Ability of Retrieval-Enhanced Transformers** (✓2302.12128) —
  RETRO'nun kazancının ne kadarının gerçek genellemeden geldiğini sorgular; senin S24-#3
  ablation'ının gerekçesi.

## IV.2 Bilgi externalizasyonu (en yakın önceki çalışma)
- **Memory³: Language Modeling with Explicit Memory** (Yang et al., 2024, ✓2407.01178) —
  **senin fikrinin en yakın akrabası.** Explicit memory'yi parametre ve RAG arasında üçüncü
  bellek formu olarak konumlandırır; metni attention-benzeri KV çiftlerine çevirip **diskte**
  saklar; **memory sparsification** ve **iki aşamalı pretraining** ile 2.4B'lik bir modeli
  sıfırdan eğitir; daha büyük LLM'leri ve RAG'ı geçer, RAG'dan hızlı decode eder.
  → *Önerinin 1., 3. ve 5. fikirleri büyük ölçüde burada.*
- **Memory Layers at Scale** (Berges et al., Meta FAIR, 2024, ✓2412.09764) — FLOP artırmadan
  trainable key-value lookup; 128B memory parametresine kadar scaling law; **1.3B + memory ≈
  Llama-2-7B** factual QA'de. → *Senin “daha az aktif parametre ile aynı bilgi” iddianın
  bugünkü en iyi kanıtı — ve tavanı.*
- **Product-Key Memory** (Lample et al., 2019), **PEER / Mixture of a Million Experts**
  (He, 2024) — seyrek büyük memory'nin verimli indekslenmesi.
- **MLP Memory** (✓2508.01832) — retriever'ı taklit etmek üzere pretrain edilmiş MLP;
  non-differentiable retrieval'ı bypass etme yaklaşımı.

## IV.3 Test-time öğrenen bellek
- **Titans** (Behrouz, Zhong, Mirrokni, Google Research, NeurIPS 2025, ✓2501.00663) —
  derin nonlineer recurrent **neural long-term memory module**; forward pass sırasında kendi
  ağırlıklarını optimize eder; gradyan-tabanlı **“surprise”** metriği + momentum; adaptive
  forgetting ile taşmayı önler. → *Senin “memory reasoning loop'unun parçası olmalı” fikrinin
  en gelişmiş hali; ama context sıkıştırma için, fact externalization için değil.*
- **Neural Turing Machines / DNC** (Graves et al.) — öğrenilmiş okuma/yazma adresleme; tarihsel kök.

## IV.4 Latent / recurrent reasoning
- **Universal Transformers** (Dehghani et al., 2018) — ağırlık paylaşımlı derinlik + ACT halting.
- **Huginn / Scaling up Test-Time Compute with Latent Reasoning** (Geiping et al., 2025,
  ✓2502.05171) — recurrent bloğu test-time'da keyfi derinliğe açar; özel veri gerektirmez,
  küçük context ile çalışır, kelimeye dökülemeyen reasoning'i yakalar.
  **Kritik sınır:** performans belirli bir iterasyon derinliğinde tepe yapıp sonra keskin düşer.
- **Stabilizing Recurrent Dynamics in Looped LMs** (✓2605.26733) — yukarıdaki instabiliteye
  doğrudan yanıt; Faz 5'te referans al.
- **Hierarchical Reasoning Model** (✓2506.21734), **A Survey on Latent Reasoning** (✓2507.06203),
  **Coconut** (Hao et al., 2024) — latent uzayda düşünme ailesi.

## IV.5 Bilgi–reasoning ayrışması (hipotezinin bilimsel zemini)
- **Physics of LM 3.3: Knowledge Capacity Scaling Laws** (Allen-Zhu & Li, ✓2404.05405) —
  **parametre başına 2 bit** (int8'de bile); 7B → 14 Gbit. → *Hem parametre aritmetiğinin
  (Hata 1) hem capacity starvation'ın (S19) temeli.*
- **Procedural Knowledge in Pretraining Drives Reasoning** (Ruis et al., ✓2411.12580) —
  influence functions ile: factual sorular ayrık dokümanlardan beslenir, reasoning soruları
  **aynı task içinde paylaşılan prosedürel dokümanlardan**; kod verisinin etkisi orantısız
  büyük. → *Ayrışma hipotezinin en güçlü ampirik dayanağı.*
- **Thinking to Recall** (Google Research) — reasoning trace üretmek parametrik bilgiyi “açar”:
  computational buffer + **factual priming**. → *Ayrışma hipotezinin en güçlü karşı-kanıtı.*
- **Disentangling Memory and Reasoning Ability in LLMs** (✓2411.13504).
- **Data Distributional Properties Drive Emergent ICL** (Chan et al., NeurIPS 2022, ✓2205.05055) —
  burstiness, çok sayıda nadir sınıf, **dinamik item anlamı** → ICL; in-weights learning ile
  **takas ilişkisinde**, ama **Zipfian çarpıklıkta bir arada var olabiliyorlar**.
  → *Senin sentetik değişken evrenler fikrinin teorik gerekçesi ve tasarım reçetesi.*

## IV.6 Retrieval vs parametrik bilgi: ölçüm ve çatışma
- **To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining** (✓2604.00715,
  COLM 2026) — OLMo-2 tabanlı 30M–3B, 100B token DCLM; pretraining korpus ölçeği (1–150x)
  × retrieval store ölçeği (1–20x); **3-boyutlu scaling çerçevesi**; retrieval tüm ölçeklerde
  parametric-only'yi geçiyor. → *Kol B'nin protokol referansı.*
- **Quantifying Prior Dominance in RAG Systems** (✓2606.23695) — **NCU** metriği (log-prob
  tabanlı, zero-shot/oracle/adversarial); ticari bir API çelişkilerin **%47.1'inde** açık
  talimata rağmen parametrik prior'a dönüyor; “epistemic blindness”. Küçük modeller katı
  factual extraction'da büyükleri yakalıyor veya geçiyor.
- **Does RAG Know When Retrieval Is Wrong?** (✓2605.14473), **Resisting Contextual Interference
  in RAG** (✓2506.05154), **Entity-Based Knowledge Conflicts in QA** (Longpre et al., 2021).
- **Reusing Pre-Training Data at Test Time is a Compute Multiplier** (✓2511.04234).

## IV.7 Temsil ve sıkıştırma
- **Toy Models of Superposition** (Elhage et al., 2022) — özelliklerin boyuttan fazla sayıda,
  yaklaşık-ortogonal paketlenmesi. → *“Aynı latent farklı anlamlar taşısın” fikri zaten gerçek.*
- **HyperNetworks** (Ha et al., 2016), **FiLM** (Perez et al., 2018) — context-koşullu ağırlık üretimi.
- **DiskANN** (NeurIPS 2019), **SPANN** (2021) — milyar ölçekli disk-tabanlı ANN;
  sorgu başına az sayıda rastgele okuma.
- **ROME** (2202.05262), **MEMIT** (2210.07229) — parametrik knowledge editing; externalization'ın
  çözmeye çalıştığı problemin mevcut, kırılgan alternatifi.

## IV.8 Vizyon
- **Karpathy, “cognitive core”** (2025) — ansiklopedik bilgiyi kabiliyet için maksimum feda eden
  birkaç milyar (hatta ~1B) parametrelik, her cihazda sürekli çalışan model; natively multimodal,
  matryoshka kapasite ayarı, agresif tool kullanımı, on-device LoRA slotları.
  → **Senin 1. fikrinle neredeyse birebir aynı. Vizyon novelty değil; mekanizma ve ölçüm novelty.**

## IV.9 Doğrulama notu
✓ işaretli arXiv ID'leri bu oturumda doğrulandı. İşaretsiz olanlar isim/yazar/yıl ile verildi;
ID'lerini yazmadan önce teyit et.

---

# Part V — Critical Problems

Önem sırasıyla, her biri için **erken tespit sinyali** ve **azaltma stratejisi**.

### P1. Reasoning, bilgi çıkarıldığında hayatta kalmayabilir  🔴
Ruis et al. reasoning'in prosedürel dokümanlardan beslendiğini gösteriyor — ama o dokümanlar
somut örnekler, örnekler de bilgi taşıyor. *Thinking to recall* ise CoT'un parametrik bilgiyi
**açtığını** (factual priming) gösteriyor; zemini çekersen mekanizma boşta kalır.
- **Sinyal:** Faz 3 sonunda GSM8K/ARC-Challenge/BBH düşüşü > %15.
- **Azaltma:** Prosedürel veriyi (kod, çalışılmış çözümler) core'da **agresif biçimde tut** —
  externalize edilecek şey [3] katmanı (keyfi atom), [1] değil. Externalizability spectrum'u
  erken ölç, sınırı bul, orada dur.

### P2. Model yine de ezberler  🔴
Prior dominance %47.1. Memory eklemek modelin onu kullanacağı anlamına gelmiyor.
- **Sinyal:** parametric leakage > 0.3; counterfactual fidelity < 0.5.
- **Azaltma:** Capacity starvation (birincil, bilgi-teorik garanti) + anti-memorization loss +
  epoch başına değer yeniden örnekleme + entity aliasing. Dördünü birden uygula.

### P3. Latent depolama patlaması  🔴
Tüm katman KV: metnin ~8000x'i. 100 GB metin → 800 TB.
- **Sinyal:** Faz 0'da store metnin 5x'inden büyük.
- **Azaltma:** **Yalnızca** PQ'lu chunk-summary (~32 vektör/chunk, 64 B/vektör). Per-token KV'yi
  tasarımdan tamamen çıkar. `s` (chunk başına vektör) ve PQ bit genişliğini ablate et.

### P4. Drift  🟠
Core güncellendikçe latent'ler geçersizleşir; TB'ları yeniden encode etmek imkânsız.
- **Sinyal:** checkpoint mesafesi arttıkça retrieval recall düşüyor.
- **Azaltma:** Hibrit depolama + versiyon etiketi + lazy re-encode + read-adapter + key/query
  normalization. **Drift eğrisini ölç ve yayınla.**

### P5. Retriever–LM co-adaptation instability  🟠
Index modeli kovalar (REALM'in klasik problemi); retriever ile LM birbirini bozar.
- **Sinyal:** eğitim sırasında recall salınımı, loss'ta periyodik sıçramalar.
- **Azaltma:** Faz 1'de encoder'ı ayrı eğit; Faz 3'te düşük LR + periyodik (her N adımda) index
  yenileme; “frozen vs co-trained” ablation'ı (#16) ile erken karar ver.

### P6. Kapı öğrenilemez  🟠
Top-k non-differentiable; gate ya hep açık ya hep kapalıya çöker.
- **Sinyal:** gate entropisi → 0; retrieval oranı 0 veya 1'e yapışıyor.
- **Azaltma:** Oracle-gain supervised eğitim (RL'den kaçın); entropy bonus; oracle kapıyı
  üst sınır olarak raporla.

### P7. Recurrence doyar veya ıraksar  🟠
Huginn: tepe-sonra-keskin-düşüş. Latent döngüde ayrıklaştırma yok → hata birikir.
- **Sinyal:** derinlik–doğruluk eğrisi monoton değil.
- **Azaltma:** Tam weight-tying yerine 2–4 farklı blok; rastgele unroll derinliği ile eğitim;
  truncated BPTT; adaptive halting'i **ikinci** fazda ekle; stabilizasyon literatürünü uygula.

### P8. Paylaşılan basis kazancı tam yanlış yerde sıfır  🔴
`I(M;C) ≈ 0` keyfi long-tail fact'ler için — yani tam da externalize etmek istediğin şeyler için.
- **Sinyal:** codebook/Context-Interpreter ablation'ı (#20) fark yaratmıyor.
- **Azaltma:** Bunu bir başarısızlık değil **bulgular** olarak çerçevele. Compositional
  compression'ı semantik iddia olmaktan çıkarıp PQ olarak konumlandır (orada kazanç gerçek ve
  ölçülebilir: 8–32x).

### P9. Değerlendirme confound'u  🔴
Kazanç externalization'dan değil, sisteme eklenen toplam parametre/hesaptan geliyor olabilir.
- **Sinyal:** iso-total-bytes karşılaştırmasında avantaj kayboluyor.
- **Azaltma:** **4 yönlü iso-tablo zorunlu** (S16). Ayrıca “rastgele içerikli memory” (#3) ve
  “prompt-level RAG” (#2) kontrolleri. Bunlar olmadan hiçbir sayı yayınlanabilir değil.

### P10. Latency  🟠
T reasoning adımı × K retrieval; recurrent döngü serileştirilmiş.
- **Sinyal:** tok/s < dense baseline'ın %30'u.
- **Azaltma:** Hiyerarşik retrieval (S11) + speculative prefetch + hard budget cap +
  retrieval cache. Inference'ta SSD darboğaz değil (S10) — asıl maliyet ANN CPU ve serileştirme.

### P11. Sentetik → doğal transfer boşluğu  🟠
- **Sinyal:** sentetik suite'te yüksek, PopQA'da kazanç yok.
- **Azaltma:** Sentetik fact'leri gerçek metin akışının içine göm; %40–50 doğal metin karıştır;
  transfer'i ayrı bir metrik olarak raporla.

### P12. Memory miss'te daha kötü halüsinasyon  🟠
Retrieval'a güvenmeyi öğrenmiş ama boş dönüşü işlemeyi öğrenmemiş model, dense baseline'dan
daha kötü olabilir.
- **Sinyal:** eksiklik ablation'ında (#19) hata oranı baseline'ı aşıyor.
- **Azaltma:** Eğitim karışımına %5–10 “fact memory'de yok” örneği; abstention'ı ödüllendir;
  derivation gain'i ayrı ölç.

---

# Part VI — Experimental Plan

## VI.1 İki kol

```
KOL A — MEKANİZMA (1B, 2 hafta)          KOL B — BİLİM (50M–400M sıfırdan, 6 hafta)
├─ OLMo-2-1B Base, core donuk             ├─ Kontrollü korpus, capacity starvation
├─ Memory modülleri eğitilir              ├─ Eşleştirilmiş dense vs core+memory
├─ Soru: memory loop'a bağlanıyor mu?     ├─ Soru: eğri EĞİMİ farklı mı?
└─ Metrik: counterfactual fidelity        └─ Metrik: PER, bits/active-param, leakage
   ⚠ tasarruf iddiasını TEST EDEMEZ           ✅ ana bilimsel sonuç buradan gelir
```

## VI.2 Kol B deney matrisi

| Eksen | Değerler | Koşu sayısı |
|---|---|---|
| Core ölçeği | 50M / 150M / 400M | 3 |
| Kol | dense baseline / core+memory | 2 |
| Memory ölçeği (treatment) | 1x / 5x / 20x fact set | 3 |
| Anti-memorization λ | 0 / 0.1 | 2 |

Tam faktöriyel pahalı; **kademeli tasarım**:
1. **Aşama 1 (6 koşu):** 3 ölçek × {dense, core+memory@5x}, λ=0.1 → ana eğri karşılaştırması.
2. **Aşama 2 (4 koşu):** en iyi ölçekte memory ölçeği süpürmesi (1x/20x) ve λ∈{0, 0.1}.
3. **Aşama 3:** ablation'lar (Part VI.4), en iyi konfigürasyonda tek ölçekte.

Her koşu için sabit token bütçesi (iso-compute) **ve** ayrıca sabit veri bütçesi — ikisi farklı
sonuç verirse bu kendi başına raporlanacak bir bulgudur.

## VI.3 Kontrollü korpus spesifikasyonu

```
Entity'ler      : 10⁵–10⁸ rastgele ID ("QX-4471"), gerçek isim YOK
Attribute'lar   : entity başına 5–20; sayısal, kategorik, ilişkisel
Evrenler        : U ∈ {1, 10, 100, 1000} — attribute değerleri evrenler arası yeniden örneklenir
Prosedürel metin: türetmeler, hesaplar, kod, çok-adımlı akıl yürütme — EVRENLER ARASI SABİT
Doğal metin     : %40–50 dekontamine Dolma/DCLM dilimi
Çelişki         : %10–15 memory ≠ prior
Eksiklik        : %5–10 fact memory'de YOK ama türetilebilir / türetilemez (ikisi de)
Dağılım         : Zipfian çarpıklık + burstiness  (Chan et al. reçetesi)
Held-out        : evren düzeyinde tutma (entity düzeyinde değil)
```

**Capacity starvation kalibrasyonu:**
```
fact_set_bits ≥ 10 × (2 bit × core_params)
```
150M core → 300 Mbit tavan → fact set ≥ 3 Gbit (≈ 60M fact × 50 bit). Bunu Faz 0'da doğrula.

## VI.4 Ablation öncelik sırası

**Tier 1 — kesilemez (iddiayı taşıyanlar):**
`#2 prompt-RAG kontrolü` · `#3 rastgele memory` · `#4 memory ablate` ·
`#9 anti-memorization λ` · `#10 evren çeşitliliği` · `#19 eksiklik`

**Tier 2 — tasarım kararları:**
`#6 granülarite` · `#7 entegrasyon noktası` · `#8 memory temsili` · `#13 controller` · `#17 memory ölçeği`

**Tier 3 — ikincil katkılar:**
`#11 starvation oranı` · `#12 recurrence derinliği` · `#14 write-back` · `#15 drift` ·
`#16 frozen vs co-trained` · `#18 çelişki/NCU` · `#20 Context Interpreter + β terimi`

## VI.5 Go / No-Go kapıları

| Kapı | Zaman | Kriter | Başarısızlıkta |
|---|---|---|---|
| **G0** | Hafta 2 | Store boyutu ≤ metnin 2x'i | Temsili yeniden tasarla (P3) |
| **G1** | Hafta 4 | Kol A: counterfactual fidelity > 0.5 | Plumbing hatalı; mimariye geçme |
| **G2** | Hafta 7 | Kol B: parametric leakage < 0.15 | Starvation oranını 10x artır (P2) |
| **G3** | Hafta 9 | Eğri eğimi farkı istatistiksel anlamlı | **Güçlü hipotez ölü** → zayıf forma çekil, ölçüm makalesi yaz |
| **G4** | Hafta 10 | Reasoning düşüşü < %15 | Externalization sınırını raporla (C3), ilerletme |
| **G5** | Hafta 12 | iso-total-bytes'ta avantaj korunuyor | P9 gerçekleşti; iddiayı daralt |

**G3 ve G4'ün başarısızlığı projeyi bitirmez** — C2 ve C3 katkıları negatif sonuçla da
yayınlanabilir. Bu, planın en değerli özelliği: her yolda bir çıktı var.

---

# Part VII — Training Plan

## VII.1 Faz tablosu

| Faz | Süre | Eğitilen | Donuk | Veri | Çıkış kriteri |
|---|---|---|---|---|---|
| **0** Store inşası | 1 hf | — | — | Korpus + dekontaminasyon | G0: store ≤ 2x metin |
| **1** Encoder | 1 hf | Memory Encoder, PQ | Core | Chunk rekonstrüksiyon + contrastive | recall@10 > 0.9 |
| **2** Read-path warmup | 1 hf | Cross-Attn, Query Head, Gate | Core, Encoder | Memory-bağımlı sentetik | G1: CF > 0.5 |
| **3** Continued pretrain | 3 hf | Hepsi (core düşük LR) | — | Tam karışım + `λ_mem` | G2: leakage < 0.15; PPL bozulması < %3 |
| **4** Controller | 1 hf | Gate/Controller | Diğerleri | Oracle-gain etiketleri | oracle'ın %90'ı, retrieval %50 ↓ |
| **5** Recurrence | 2 hf | Recurrent Block, Halting | — | Rastgele unroll derinliği | Derinlik eğrisi monoton |
| **6** Write-back | 1 hf | Trust Head, policy | — | Türetme görevleri | Derivation gain > 0; hatalı yazma < %1 |
| **B** Scaling (paralel) | 6 hf | Sıfırdan 50M/150M/400M | — | Kontrollü korpus | G3, G5 |

## VII.2 Kritik eğitim detayları

**Zero-init gating.** Her yeni cross-attention bloğu `h ← h + γ·CrossAttn(...)` biçiminde, `γ`
sıfırdan başlar. Böylece Faz 2'nin başında model tam olarak pretrained core'a denktir ve
bozulma olmaz. (ReZero / LoRA'nın standart hilesi; atlanırsa Faz 2'de PPL patlar.)

**Anti-memorization loss'un pratik uygulaması.** Her batch'in rastgele bir alt kümesini
(%10–20) memory olmadan ikinci kez forward et; o forward'da cevap tokenlarının entropisini
**ödüllendir**. Maliyet: ~%15 ekstra compute. `λ_mem`'i 0'dan başlatıp warmup ile yükselt —
baştan yüksek verirsen model hiçbir şey öğrenemez.

**Dejenerasyon kontrolü.** `λ_mem` çok yüksekse model memory *varken de* belirsizleşir.
Her değerlendirmede `acc(memory ✓)` ve `acc(memory ∅)` **birlikte** raporla; sağlıklı
sonuç ikisi arasında büyük fark, ikisinde de yüksek/düşük değil.

**Index yenileme.** Faz 3'te encoder de güncelleniyorsa index'i her N adımda yenile
(REALM'in asenkron yaklaşımı). N'i ablate et; çok seyrek → stale, çok sık → maliyet ve instabilite.

**Recurrence eğitimi.** Faz 5'te unroll derinliğini her batch'te rastgele örnekle (örn. 1–16);
truncated BPTT ile son k adımdan geriye yay. Halting head'i önce **denetimli** eğit
(zorluk etiketiyle), adaptive halting'i en sona bırak.

## VII.3 Compute bütçesi

```
Kol A (1B, donuk core, adaptör eğitimi)
  ~3B token × 200M eğitilebilir  →  4060'ta ~1 hafta   veya  A100'de ~8 saat

Kol B (sıfırdan scaling sweep)
  150M × 10B token = 6ND = 9e18 FLOP
  H100 @ ~40% MFU (~1.6e14 FLOP/s) → ~15.6 saat → ~$30–60/koşu
  Aşama 1+2 = 10 koşu (ölçekler farklı maliyette) → ~$300–600 toplam

Ablation'lar: en küçük ölçekte (50M), koşu başına ~2 saat → ~$5–10
```

**Toplam bulut bütçesi: ~$600–1200.** Projenin bilimsel omurgası bu parayla satın alınıyor.
Yerel 4060 geliştirme, debug, inference ve küçük ablation'lar için.

---

# Part VIII — Evaluation Plan

## VIII.1 Metrik hiyerarşisi

**Seviye 1 — Externalization gerçekten oldu mu? (senin özgün katkın)**
```
Parametric leakage      L  = acc(memory ∅) − acc(random)          ↓ hedef < 0.15
Memory dependence       D  = acc(memory ✓) − acc(memory ∅)        ↑ hedef > 0.6
Counterfactual fidelity CF = P(cevap yeni değere döner | fact değişti)  ↑ hedef > 0.85
NCU                        (log-prob tabanlı context kullanımı)     ↑
Prior dominance            (çelişkide prior'a dönme oranı)          ↓ hedef < %10
```

**Seviye 2 — Verimlilik iddiası**
```
PER (parameter-equivalence ratio)   : dense baseline hangi boyutta eşitleniyor
bits-per-active-parameter           : Allen-Zhu yöntemiyle
Derivation gain                     : acc(yok ama türetilebilir) − acc(yok ve türetilemez)
```

**Seviye 3 — Yan hasar**
```
Reasoning         : GSM8K, MATH, ARC-Challenge, BBH, ProntoQA
Dil               : held-out PPL — knowledge-dense vs procedure-dense dilimlere AYRILMIŞ
Kalibrasyon       : memory miss'te abstention oranı, halüsinasyon oranı
```

**Seviye 4 — Sistem maliyeti**
```
aktif parametre · toplam bayt (VRAM+RAM+SSD) · FLOPs/token · tok/s ·
retrieval/token · IOPS/token · p50/p99 latency · J/token
```

## VIII.2 Zorunlu iso-karşılaştırma tablosu

| | iso-active-param | iso-FLOP | **iso-total-bytes** | iso-latency |
|---|---|---|---|---|
| Dense baseline | | | | |
| Prompt-level RAG | | | | |
| CC-XM (bizim) | | | | |
| CC-XM, rastgele memory | | | | |

**Dört sütun da doldurulacak.** Literatürdeki memory/retrieval makalelerinin çoğu birini
seçip diğerlerini atlıyor; dördünü birden raporlamak metodolojik bir katkıdır ve
P9 confound'una karşı tek savunmadır.

## VIII.3 Benchmark seti

| Kategori | Setler |
|---|---|
| Long-tail bilgi | **PopQA**, EntityQuestions |
| Açık-alan QA | NQ-open, TriviaQA |
| Multi-hop | HotpotQA, **MuSiQue** |
| Reasoning (yan hasar) | GSM8K, MATH, **ARC-Challenge**, BBH |
| Sentetik mantık | ProntoQA / ProofWriter |
| Güncellenebilirlik | Temporal QA, FreshQA tarzı |
| Çelişki / memory kullanımı | **Kendi counterfactual-worlds suite'in** + NCU |
| LM kalitesi | Held-out PPL, dilimlere ayrılmış |
| *(ikincil)* | MMLU — **birincil yapma**, kontamine ve iki yeteneği karıştırıyor |

## VIII.4 Yayınlanacak artefaktlar

1. **Counterfactual-worlds benchmark** (korpus üreteci + değerlendirme scripti) — C2'nin somut çıktısı.
2. **Externalizability spectrum tablosu** — bilgi sınıfı × externalization kaybı.
3. **Drift eğrisi** — checkpoint mesafesi × retrieval recall / task accuracy.
4. Tüm iso-karşılaştırma tabloları, ham sayılarla.

---

# Part IX — Hardware Requirements

## IX.1 Mevcut donanım: RTX 4060 8 GB + 32 GB RAM

```
RTX 4060 : 8 GB GDDR6, ~272 GB/s, PCIe 4.0 x8 (~16 GB/s)
DDR5     : ~60–90 GB/s (dual channel)
NVMe Gen4: ~7 GB/s sıralı, ~100 µs rastgele okuma (QD1), ~1M IOPS (yüksek QD)
```

| İş | Durum | Detay |
|---|---|---|
| 1B core donuk + memory modülü eğitimi | ✅ | bf16; 2 GB donuk ağırlık; grad checkpointing; seq 1024; micro-batch 1–2 + accumulation |
| 1B/3B 4-bit inference + memory | ✅ Rahat | ~0.6 / 1.8 GB ağırlık |
| RAM'de ANN index | ✅ | **100M vektör × 64 B PQ = 6.4 GB** — PoC'de SSD'ye gerek bile yok |
| SSD memory store (PoC) | ✅ | ~20 GB PQ'lu chunk-summary |
| Sıfırdan ≤200M eğitimi | 🟡 | 150M × 10B token ≈ **5 gün** |
| Kol B tam sweep | ❌ | Aylar sürer → **kirala** |
| 7B ölçekleme | ❌ | VRAM yetmez |

## IX.2 Hedef inference bütçesi (uzun vade)

```
VRAM  4–8 GB   : Cognitive Core (4-bit) + KV cache + working set (k=8..64 latent)
RAM  16–32 GB  : Kaba ANN index + recent memory + retrieval cache (~100M PQ vektör)
SSD  0.1–2 TB  : Ground-truth metin + PQ'lu latent cache + derived-fact cache
```

## IX.3 Bulut ihtiyacı

| Kalem | Tahmin |
|---|---|
| Kol B Aşama 1+2 (10 koşu) | ~$300–600 |
| Ablation'lar (50M ölçeğinde) | ~$150–300 |
| Yedek / yeniden koşular | ~$150–300 |
| **Toplam** | **~$600–1200** |

## IX.4 Dürüst tavan (S28)

> **8 GB VRAM'de, ~2B aktif parametre ile 20–30B dense sınıfı *bilgi* performansı +
> yeniden eğitim gerektirmeyen güncellenebilir bilgi tabanı.**

Bu savunulabilir. *“Frontier in 8 GB”* savunulabilir değil — frontier avantajı artık bilgi
değil, reasoning ve post-training; externalization o boşluğu kapatmıyor.

---

# Part X — 3-Month Prototype Roadmap

## Ay 1 — Altyapı ve erken öldürme testleri

| Hafta | İş | Çıktı | Kapı |
|---|---|---|---|
| **1** | Korpus üreteci (evrenler, entity aliasing, Zipfian+burstiness); dekontaminasyon | Kontrollü korpus v1 | |
| **1** | **Ölçüm altyapısı önce**: leakage, CF, D, NCU harness'ı | Eval harness | |
| **2** | Memory encoder + PQ + ANN index; store boyutu ölçümü | Memory store | **G0** ≤2x metin |
| **2** | Baseline'lar: dense, prompt-level RAG | Baseline sayıları | |
| **3** | Kol A: OLMo-2-1B + cross-attn (zero-init) + query head, core donuk | Çalışan read-path | |
| **4** | Kol A: memory-bağımlı sentetik üzerinde eğitim; CF ölçümü | Mekanizma raporu | **G1** CF>0.5 |

> **Ay 1'in tasarım kararı:** ölçüm altyapısını modelden **önce** kur. Aksi halde Ay 3'te
> çalışan bir sistemin olur ama çalışıp çalışmadığını bilemezsin.

## Ay 2 — Bilimsel deney (asıl iş)

| Hafta | İş | Çıktı | Kapı |
|---|---|---|---|
| **5** | Kol B altyapısı; capacity starvation kalibrasyonu; 50M pilot | Pilot eğri | |
| **6** | Anti-memorization loss implementasyonu + λ warmup; dejenerasyon kontrolü | λ süpürmesi | |
| **7** | Kol B Aşama 1: 3 ölçek × {dense, core+memory} | **Ana eğri** | **G2** leakage<0.15 |
| **8** | Tier-1 ablation'lar: prompt-RAG, rastgele memory, memory-ablate, evren çeşitliliği, eksiklik | Ablation tablosu | |

## Ay 3 — Sınırlar, sistem, yazım

| Hafta | İş | Çıktı | Kapı |
|---|---|---|---|
| **9** | Kol B Aşama 2 (memory ölçeği); eğim analizi, istatistiksel test | Scaling sonucu | **G3** |
| **10** | Externalizability spectrum; reasoning yan hasar ölçümü | Spectrum tablosu | **G4** <%15 |
| **11** | Controller (oracle-gain) + drift eğrisi + read-adapter; 4 yönlü iso-tablo | Sistem sonuçları | **G5** |
| **12** | Write-back / derivation gain (opsiyonel C4); makale taslağı; benchmark yayını | **Taslak + artefaktlar** | |

## Kapsam dışı bırakılanlar (bilinçli)

- **Recurrent core** — Ay 1–3'e sığmaz ve memory ile aynı anda debug edilemez. Ay 4+.
- **Context Interpreter (D(C) hypernetwork)** — en kırılgan bileşen; Hafta 11'de tek bir
  ablation olarak dene, mimariye baştan koyma.
- **7B ölçekleme** — ancak G3 geçilirse.
- **Multimodal, tool use, RL post-training** — tamamen kapsam dışı.

## Her yolda bir çıktı

```
G3 ✓ ve G4 ✓  →  "Knowledge externalization ile 3–5x aktif parametre tasarrufu" (ana makale)
G3 ✓ ve G4 ✗  →  "Externalization çalışıyor ama reasoning'in bedeli şu"  (hâlâ güçlü makale)
G3 ✗          →  "Externalization şurada doyuyor: ölçüm çerçevesi + benchmark"  (C2+C3, workshop→konferans)
G2 ✗          →  "Modeller memory varken bile ezberliyor: anti-memorization'ın sınırları" (negatif sonuç, yayınlanabilir)
```

---

# Part XI — Research-Paper-Worthy Contribution

## XI.1 Önerilen makale

> **“Forcing Externalization: Training Objectives and Measurements for Separating
> Factual Knowledge from Reasoning in Language Models”**

**Abstract iskeleti:**
Önceki çalışmalar (Memory³, Memory Layers) bilginin model parametrelerinden dışarı
*çıkarılabileceğini* gösterdi. Ancak modellerin bunu yapmaya *zorlanıp zorlanamayacağı*,
bilginin *ne kadarının* çıkarılabileceği ve reasoning'in *nerede bozulduğu* ölçülmedi.
Biz (i) parametrik depolamayı doğrudan cezalandıran bir eğitim hedefi, (ii) externalization'ın
gerçekleşip gerçekleşmediğini ölçen bir metrik ailesi ve public benchmark, (iii) bilgi
sınıflarına göre externalization sınırının ampirik haritasını sunuyoruz.

## XI.2 Katkılar

| | Katkı | Neden yeni | Negatif sonuçta da geçerli mi |
|---|---|---|---|
| **C1** | **Anti-memorization objective** + capacity-starvation protokolü | Tüm önceki çalışmalar memory ekleyip kullanılmasını *umuyor*; kimse ezberlemeyi loss'a koymadı | ✅ |
| **C2** | **Ölçüm çerçevesi**: parametric leakage, memory dependence, counterfactual fidelity, derivation gain + counterfactual-worlds benchmark | Alan bu aracı istiyor (prior dominance literatürü boşluğu gösteriyor); kimse yapmıyor | ✅ |
| **C3** | **Externalizability spectrum** — bilgi sınıfı × kayıp haritası, reasoning kırılma noktası | Ayrışmanın *derecesi* hiç ölçülmedi | ✅ |
| **C4** | *(opsiyonel)* **Amortized derivation**: derive-or-retrieve routing + derived-fact write-back | Gerçekten yeni; "türet, ezberleme" fikrini mekanizmaya çevirir | ❌ çalışması gerekir |
| **C5** | *(ikincil)* Bütçe-farkında öğrenilmiş 3-katmanlı controller; latent drift eğrisi + read-adapter | Yarı-yeni, sağlam sistem katkıları | 🟡 |

## XI.3 Neden bu paketleme

**Mimariyi başa koyma.** Mimari novelty'n yok (Part III) ve hakem bunu ilk paragrafta bulur.
Objective + measurement'ı başa koyduğunda:

- Prior art savunulabilir hale gelir — sen Memory³ ile *yarışmıyorsun*, onun açık bıraktığı
  soruyu cevaplıyorsun.
- **Her deney sonucu yayınlanabilir.** Mimari iddiası çalışmazsa elinde hiçbir şey kalmaz;
  ölçüm çerçevesi çalışmasa bile çalışmadığını *göstermek* sonuçtur.
- Benchmark ve metrikler **atıf toplar** ve alanın diline girer. Uzun vadede en yüksek
  getirili katka türü budur.

## XI.4 Hedef mekânlar

| Mekân | Uygunluk |
|---|---|
| **COLM** | En iyi eşleşme — LM-merkezli, ampirik, benchmark katkılarına açık |
| **ICLR / NeurIPS** | C1+C3 güçlü çıkarsa |
| **NeurIPS D&B track** | C2 (benchmark) tek başına |
| **ICLR MemAgents / workshop** | Erken sonuçlar, Ay 3 çıktısı için iyi bir ilk durak |

---

# Ek A — Tek sayfalık özet

**Sağlam olan:** Semi-parametric yön doğru ve kanıtlanmış. Reasoning/knowledge ayrışmasının
veri düzeyinde temeli var (Ruis et al.). Sentetik değişken evrenlerin teorik gerekçesi var
(Chan et al.). Recurrent core consumer VRAM'de gerçekten mantıklı.

**Kırılan üç yer:**
1. **Parametre aritmetiği.** 2 bit/param yasası gereği 500 GB'ın ansiklopedik kısmı zaten
   birkaç GB. Gerçekçi tavan 3–5x, 50–100x değil.
2. **`Meaning = f(M,C)` bilgi-teorik olarak `I(M;C)` ile sınırlı** — ve tam da externalize
   etmek istediğin keyfi long-tail fact'lerde bu ≈ 0. 2. ve 3. fikrin aynı fikir, ve dağılımın
   farklı bölgelerine hitap ediyorlar.
3. **Dondurulmuş pretrained core tasarruf iddiasını test edemez.** Bilgi zaten içeride.
   Sıfırdan, küçük ölçekte, kontrollü korpus şart.

**Yapılması gereken en önemli üç şey:**
1. Ölçüm altyapısını modelden **önce** kur (leakage, counterfactual fidelity).
2. **Capacity starvation** ile ezberlemeyi bilgi-teorik olarak imkânsız kıl.
3. Bilimsel sonucu **Kol B'den** (sıfırdan 50M–400M) al, 1B adaptasyonundan değil.

**Manşet değiştir:** *"Frontier in 8 GB"* → *"knowledge-per-active-parameter Pareto'su +
editable knowledge"*. İkincisi hem doğru hem ticari olarak daha değerli.

**Ekle:** derived-fact write-back (en özgün fikrin, önerinde sadece ima edilmiş).
**Çıkar (manşetten):** context-dependent multi-meaning — PQ olarak yeniden çerçevele.
