# DDI-Ben pilotları ve kilit bulgu doğrulamaları

Önerilen değerlendirme makalesinin (bkz. `../RAPOR_2026.md` §7–8) dayandığı pilot betikleri ve bunların bağımsız yeniden uygulamaları.

## Gereken

- DDI-Bench kodu ve verisi: `git clone https://github.com/LARS-research/DDI-Bench DDI-Bench && git -C DDI-Bench checkout dfbeeab`. Bu klasörün içine klonlanmalı. Betiklerdeki yollar `DDI-Bench/...` biçiminde görelidir.
- Çıktı klasörleri: `mkdir -p out/phase2/{evalsci,methodC,synth} out/phase3/anchor_reverify`.
- numpy, scipy, scikit-learn, pandas, rdkit.
- `phase2_methodC/` CROssBAR v1 hedef listesini `../../data/drugs.json`'dan okur.

Betikler bu klasörden çalıştırılır; bazıları DDI-Bench kökünü ve bölmeyi argüman olarak alır (`python verify_skeptic/drugbank_me_indep.py <DDI-Bench/DDI_Ben/DDI_Ben> drugbank_random`).

## İçerik

| Klasör | Ne |
|---|---|
| `phase2_evalsci/` | `pilot_typeprior.py`: DrugBank-86 eğitimsiz ana-etki (ME-kNN) tabanı. `pilot_twosides_degree.py`: TWOSIDES partner etiket-derecesi tabanı. `pilot_swap.py`: takas testleri. JSON çıktılar. |
| `phase2_methodC/` | CROssBAR v1 hedef listelerinden hedef-kNN ilaç eğilimleri (küme/rastgele bölmeler, TWOSIDES) |
| `phase2_synth/` | Real Scene (onay-tarihi, yalnız S2) ME tabanı |
| `verify_A/`, `verify_B/` | İki bağımsız sıfırdan yeniden uygulama (varyant taramaları, partner-eşleşmeli negatifler, negatif etiket gürültüsü) |
| `verify_skeptic/` | Benchmark değerlendirici koduyla birebir karşılaştırma, negatif kurulum denetimi, bootstrap güven aralıkları |

Sonuçların özeti `../RAPOR_2026.md` §8.1'de.

Not: DDI-Bench reposunda lisans dosyası yok. Bölme dosyalarını ve türetilmiş tahminleri yeniden dağıtmadan önce yazarlardan izin alınmalı; burada yalnızca kod ve özet metrikler bulunuyor.
