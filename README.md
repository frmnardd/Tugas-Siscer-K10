[lembar_kerja_pbl_kelompok_10.md](https://github.com/user-attachments/files/34353805/lembar_kerja_pbl_kelompok_10.md)
# Lembar Kerja Kelompok 10 — Pertemuan 10: Data Preprocessing & Feature Engineering

**Identitas Kelompok:** Kelompok 10  
**Topik Studi Kasus:** Prediksi Kelulusan Tepat Waktu Mahasiswa (`data_akademik_kelulusan.csv`)

## Bagian 1: Evaluasi Model Awal (Sebelum Preprocessing)

### Catatan Output Program
* **Dimensi dataset awal:** `312 baris × 14 kolom` (`(312, 14)`)
* **Akurasi model pertama (`model0` - KNN mentah):** `0.619` (61,9%)
* **Akurasi baseline selalu menebak `"Ya"` (`y_te.mean()`):** `0.587` (58,7%)

### Pembahasan Pertanyaan 1
**Soal:** Bandingkan akurasi model pertama dengan akurasi kalau kita selalu menebak `"Ya"`. Apa artinya?

**Jawaban Kelompok 10:**
Performa model pertama (`0.619` atau `61,9%`) ternyata hampir sama dengan tebakan sederhana yang selalu memprediksi seluruh mahasiswa lulus tepat waktu (`"Ya"`, yaitu `0.587` atau `58,7%`), dengan selisih peningkatan hanya sebesar **0.032 (3,2%)**.

Hal ini menunjukkan bahwa **model awal gagal menangkap pola pembeda antara mahasiswa yang lulus tepat waktu dan yang terlambat**. Kegagalan tersebut dipicu oleh beberapa kesalahan mendasar:
* Pengisian nilai kosong menggunakan angka `0` (`fillna(0)`) mengaburkan makna asli data (contohnya mahasiswa yang datanya belum terinput dianggap memiliki IPK `0.0` atau kehadiran `0%`).
* Perhitungan jarak pada KNN tertutup oleh besarnya skala angka pada kolom `nim` ($\approx 2,4 \times 10^{13}$) dan `penghasilan_ortu` (jutaan rupiah) karena belum dilakukan *feature scaling* maupun pembuangan kolom ID.
* Baris ganda dan kesalahan input (*outlier* ekstrem) masih bercampur di dalam data latih, serta variabel kategorikal belum diikutsertakan.

## Bagian 2: Eksplorasi & Identifikasi Masalah Data

### Potongan Kode (`TODO` Bagian 2)
```python
print("Jumlah nilai kosong per kolom:")
print(df.isna().sum())
print("\nJumlah baris ganda:", df.duplicated().sum())
```

### Catatan Output Eksplorasi
* **Total baris duplikat:** `12` baris (dari 312 baris, jumlah data unik sebenarnya adalah 300 baris).
* **Rincian nilai kosong (`NaN`) pada tiap fitur:**
  * `kehadiran`: 22 data kosong
  * `jam_belajar`: 19 data kosong
  * `penghasilan_ortu`: 16 data kosong
  * `ipk_sem4`: 9 data kosong
  * `kemampuan_bahasa`: 6 data kosong
  * 9 kolom lainnya tidak memiliki nilai kosong (`0`).
* **Variasi nilai pada kolom kategorikal:**
  * `jenis_kelamin`: terpecah menjadi 8 format teks (`L`: 150, `P`: 119, `perempuan `: 10, `Perempuan`: 8, `laki`: 8, `l`: 8, `Laki-laki`: 6, `p`: 3).
  * `prodi`: terpecah menjadi 8 variasi akibat perbedaan huruf kapital (`Informatika`: 97, `Sistem Informasi`: 81, `Statistika`: 67, `Matematika`: 52, `matematika`: 6, `sistem informasi`: 5, `statistika`: 2, `informatika`: 2).
  * `kampus`: seluruh 312 baris hanya berisi satu nilai yang sama, yakni `Tembalang`.

### Pembahasan Pertanyaan 2
**Soal:** Tuliskan minimal lima masalah yang Anda temukan, beserta kolomnya.

**Jawaban Kelompok 10:**
1. **Keberadaan *Missing Values* (Data Kosong):** Ditemukan nilai `NaN` pada lima kolom, yaitu `kehadiran` (22 baris), `jam_belajar` (19 baris), `penghasilan_ortu` (16 baris), `ipk_sem4` (9 baris), dan `kemampuan_bahasa` (6 baris).
2. **Redundansi / Duplikasi Data (*Duplicate Records*):** Terdapat **12 baris data ganda** pada keseluruhan dataset yang menyebabkan ukuran sampel membengkak dari 300 menjadi 312 baris.
3. **Standarisasi Teks Tidak Seragam pada `jenis_kelamin`:** Penulisan jenis kelamin tidak baku karena memuat 8 variasi (`"L"`, `"l"`, `"laki"`, `"Laki-laki"`, `"P"`, `"p"`, `"Perempuan"`, serta `"perempuan "` dengan tambahan spasi di belakang).
4. **Perbedaan Huruf Besar-Kecil pada `prodi`:** Nama program studi tercatat ganda karena perbedaan kapitalisasi (`"Informatika"` dengan `"informatika"`, `"Sistem Informasi"` dengan `"sistem informasi"`, `"Statistika"` dengan `"statistika"`, serta `"Matematika"` dengan `"matematika"`).
5. **Kesalahan Desimal pada `ipk_sem4`:** Terdapat nilai maksimum IPK hingga **`37.30`** (serta nilai `28.3` dan `32.8`) yang melampaui skala tertinggi IPK (`4.00`) akibat salah penempatan titik desimal saat input.
6. **Angka Persentase Tidak Logis pada `kehadiran`:** Ditemukan nilai kehadiran **`850.0`** dan **`920.0`**, yang tidak mungkin terjadi karena batas atas kehadiran perkuliahan adalah `100%`.
7. **Durasi Waktu Mustahil pada `jam_belajar`:** Terdapat catatan waktu belajar mandiri sebesar **`120.0`** dan **`168.0` jam/minggu**, padahal 1 minggu penuh (7 × 24 jam) hanya berjumlah 168 jam.
8. **Fitur Tidak Berguna & Kebocoran Target (*Data Leakage*):** Kolom `nim` hanya berfungsi sebagai identitas unik mahasiswa, kolom `kampus` bersifat konstan (`"Tembalang"` pada seluruh baris), dan kolom `nilai_ta` mengandung informasi masa depan yang baru muncul saat mahasiswa lulus.

## Bagian 3: Pembersihan Data (*Data Cleaning*) & Penanganan *Outlier*

### Potongan Kode (`TODO` Bagian 3a & 3b)
```python
data = df.copy()

# 3a. Hapus baris ganda
data = data.drop_duplicates()

# 3b. Seragamkan penulisan jenis kelamin dan prodi
peta_jk = {"l": "L", "laki-laki": "L", "laki": "L", "p": "P", "perempuan": "P"}
data["jenis_kelamin"] = data["jenis_kelamin"].str.strip().str.lower().map(peta_jk)
data["prodi"] = data["prodi"].str.strip().str.title()

print(len(data), "baris")
print(data["jenis_kelamin"].value_counts())
```
* **Hasil Eksekusi:** Tersisa `300 baris` bersih dengan proporsi `jenis_kelamin`: `L = 165` dan `P = 135`.

### Hasil Deteksi *Outlier* Metode IQR (`cek_iqr`)
* `ipk_sem4` (rentang normal IQR `2.03` s.d. `4.24`): terdeteksi **5 nilai** $\rightarrow$ `[2.01, 28.3, 32.8, 37.3, 37.3]`
* `kehadiran` (rentang normal IQR `57.50` s.d. `109.50`): terdeteksi **2 nilai** $\rightarrow$ `[850.0, 920.0]`
* `jam_belajar` (rentang normal IQR `-4.50` s.d. `23.50`): terdeteksi **7 nilai** $\rightarrow$ `[24.0, 24.0, 25.0, 32.0, 120.0, 120.0, 168.0]`

### Pembahasan Pertanyaan 3
**Soal:** Dari nilai yang ditandai, mana yang jelas salah input dan mana yang masih masuk akal?

**Jawaban Kelompok 10:**
* **Kelompok Nilai yang Jelas Salah Input (*Invalid / Human Error*):**
  * **`ipk_sem4` bernilai `28.3`, `32.8`, `37.3`, dan `37.3`:** Tidak valid karena skala IPK perguruan tinggi maksimal adalah `4.00`; angka ini merupakan salah ketik dari `2.83`, `3.28`, dan `3.73`.
  * **`kehadiran` bernilai `850.0` dan `920.0`:** Tidak valid karena persentase kehadiran maksimal hanya `100%`.
  * **`jam_belajar` bernilai `120.0`, `120.0`, dan `168.0`:** Tidak valid karena mustahil mahasiswa belajar mandiri 17–24 jam penuh setiap hari tanpa tidur selama seminggu.
* **Kelompok Nilai Ekstrem yang Masih Masuk Akal (*Valid Extreme Values*):**
  * **`ipk_sem4` bernilai `2.01`:** Walaupun sedikit di bawah batas bawah IQR (`2.03`), nilai `2.01` adalah nilai IPK yang sah dan wajar dialami mahasiswa yang mengalami kesulitan akademik, sehingga harus dipertahankan.
  * **`jam_belajar` bernilai `24.0`, `24.0`, `25.0`, dan `32.0`:** Setara dengan waktu belajar mandiri sekitar 3,5 hingga 4,5 jam per hari. Angka ini sangat wajar bagi mahasiswa yang tekun dan bukan kesalahan input.

### Potongan Kode Perbaikan *Outlier* (`TODO` Bagian 3c)
```python
# IPK di atas 4 adalah salah ketik (3.73 tertulis 37.3): bagi 10
salah_ipk = data["ipk_sem4"] > 4
data.loc[salah_ipk, "ipk_sem4"] = data.loc[salah_ipk, "ipk_sem4"] / 10

# Kehadiran di atas 100% dan jam belajar di atas 84 jam per minggu tidak mungkin: jadikan kosong
data.loc[data["kehadiran"] > 100, "kehadiran"] = np.nan
data.loc[data["jam_belajar"] > 84, "jam_belajar"] = np.nan

data[["ipk_sem4", "kehadiran", "jam_belajar"]].describe().round(2)
```

### Ringkasan `describe()` Pasca-Koreksi *Outlier*
| Metrik | `ipk_sem4` | `kehadiran` | `jam_belajar` |
|---|---:|---:|---:|
| **count** | 292.00 | 276.00 | 279.00 |
| **mean** | 3.13 | 82.92 | 10.03 |
| **std** | 0.38 | 8.75 | 5.32 |
| **min** | 2.01 | 63.00 | 0.00 |
| **25%** | 2.86 | 77.00 | 6.00 |
| **50%** | 3.13 | 83.00 | 9.00 |
| **75%** | 3.38 | 90.00 | 13.00 |
| **max** | 4.00 | 100.00 | 32.00 |

## Bagian 4: Seleksi Fitur (*Feature Selection*)

### Potongan Kode (`TODO` Bagian 4)
```python
kolom_dibuang = ["nim", "kampus", "nilai_ta"]
kolom_numerik = ["ipk_sem4", "sks_lulus", "kehadiran", "jam_belajar", "penghasilan_ortu"]
kolom_nominal = ["jenis_kelamin", "prodi", "jalur_masuk", "ikut_organisasi"]
kolom_ordinal = ["kemampuan_bahasa"]

print("Kolom yang dibuang:", kolom_dibuang)
X = data[kolom_numerik + kolom_nominal + kolom_ordinal]
y = (data["lulus_tepat_waktu"] == "Ya").astype(int)
print(X.shape)
```
* **Hasil Eksekusi:** `Kolom yang dibuang: ['nim', 'kampus', 'nilai_ta']`, ukuran matriks fitur `X` menjadi `(300, 10)`.

### Pembahasan Pertanyaan 4
**Soal:** Jelaskan alasan setiap kolom yang dibuang. Mengapa `nilai_ta` berbahaya walaupun sangat berkaitan dengan label?

**Jawaban Kelompok 10:**
* **Alasan membuang `nim`:** Kolom ini hanyalah kode penomoran administratif untuk tiap mahasiswa dan sama sekali tidak merepresentasikan performa akademik. Mempertahankan `nim` justru merusak perhitungan model karena nilainya sangat besar ($\sim 2,4 \times 10^{13}$).
* **Alasan membuang `kampus`:** Seluruh mahasiswa pada data berasal dari kampus yang sama (`"Tembalang"`). Karena nilainya seragam 100% (varians = 0), kolom ini tidak memiliki daya pisah sama sekali terhadap target kelulusan.
* **Alasan membuang `nilai_ta` & bahayanya:** Kolom `nilai_ta` memicu **kebocoran data (*target/data leakage*)**. Sistem prediksi ini dirancang untuk mendeteksi risiko ketidaklulusan ketika mahasiswa baru menyelesaikan **semester 4**, sementara nilai Tugas Akhir (`nilai_ta`) baru didapatkan pada semester akhir saat mahasiswa menempuh sidang kelulusan. Apabila `nilai_ta` dipakai, model akan tampak sangat akurat di atas kertas, tetapi **mustahil digunakan pada mahasiswa semester 4** karena nilainya belum ada.

## Bagian 5 & 6: Pembagian Data & *Pipeline Preprocessing*

### Potongan Kode Pembagian Data (`TODO` Bagian 5)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape)
```
* **Hasil Eksekusi:** `(240, 10) (60, 10)` — data terbagi menjadi 240 sampel latih dan 60 sampel uji dengan proporsi kelas lulus/tidak lulus yang konsisten (`stratify=y`).

### Potongan Kode *Pipeline* (`TODO` Bagian 6)
```python
pra_numerik = Pipeline([
    ("imputasi", SimpleImputer(strategy="median")),
    ("scaling", StandardScaler()),
])
pra_nominal = Pipeline([
    ("imputasi", SimpleImputer(strategy="most_frequent")),
    ("encoding", OneHotEncoder()),
])
pra_ordinal = Pipeline([
    ("imputasi", SimpleImputer(strategy="most_frequent")),
    ("encoding", OrdinalEncoder(categories=[["Dasar", "Menengah", "Mahir"]])),
])
preprocessing = ColumnTransformer([
    ("num", pra_numerik, kolom_numerik),
    ("nom", pra_nominal, kolom_nominal),
    ("ord", pra_ordinal, kolom_ordinal),
])
```

## Bagian 7: Pemodelan & Perbandingan Akurasi

### Tabel Hasil Pengujian Model
| Nama Model | Skor Akurasi Data Uji | Peningkatan dari Model Pertama |
|---|---:|---:|
| **KNN (Setelah Preprocessing)** | **0.833** (83,3%) | **+0.214** (+21,4 poin persentase) |
| **Regresi Logistik (Setelah Preprocessing)** | **0.800** (80,0%) | **+0.181** (+18,1 poin persentase) |
| **Model Pertama (Sebelum Preprocessing)** | **0.619** (61,9%) | — |

### Pembahasan Pertanyaan 5
**Soal:** Berapa kenaikan akurasi dibanding model pertama?

**Jawaban Kelompok 10:**
* Dibandingkan model pertama yang hanya memperoleh akurasi **0.619 (61,9%)**, model **KNN setelah preprocessing** berhasil mencapai akurasi **0.833 (83,3%)**, sehingga terdapat kenaikan akurasi sebesar **0.214 (atau meningkat 21,4 poin persentase)**.
* Sementara itu, model **Regresi Logistik setelah preprocessing** mencapai akurasi **0.800 (80,0%)**, yang berarti naik sebesar **0.181 (atau meningkat 18,1 poin persentase)** dibandingkan model pertama.

## Bagian 8: Eksperimen Tambahan

### 8a. Dampak Menghilangkan *Scaling* pada KNN
* **Hasil Eksekusi:** `KNN tanpa scaling: 0.617` (61,7%)

#### Pembahasan Pertanyaan 6
**Soal:** Mengapa KNN sangat terpengaruh oleh scaling? Kolom mana yang mendominasi perhitungan jarak?

**Jawaban Kelompok 10:**
* **Alasan KNN bergantung pada *scaling*:** Klasifikasi KNN bekerja dengan mengukur jarak antar titik data di ruang multidimensi (menggunakan rumus jarak Euclidean $\sqrt{\sum (x_{ij} - x_{kj})^2}$). Jika tiap kolom memiliki satuan yang timpang, maka selisih pada kolom berangka besar akan menghasilkan nilai kuadrat yang sangat masif, sedangkan selisih pada kolom berangka kecil (seperti `ipk_sem4` yang hanya berbeda `0.1–1.0` poin) menjadi tidak dianggap dalam penentuan tetangga terdekat.
* **Kolom yang mendominasi jarak:** Kolom **`penghasilan_ortu`**, sebab rentang nilainya berada di angka jutaan hingga puluhan juta rupiah (`1.200.000` s.d. `22.400.000`). Tanpa `StandardScaler()`, jarak antar mahasiswa hampir seluruhnya ditentukan oleh selisih penghasilan orang tua mereka, bukan oleh IPK, SKS, maupun kehadiran, sehingga akurasi KNN merosot tajam dari **0.833** kembali ke **0.617**.

### 8b. Dampak Menyertakan Kolom `nilai_ta`
* **Hasil Eksekusi:** `Akurasi dengan nilai_ta: 1.0` (100%)

#### Pembahasan Pertanyaan 7
**Soal:** Akurasinya sangat tinggi. Mengapa hasil ini tidak bisa dipakai untuk memprediksi mahasiswa semester 4?

**Jawaban Kelompok 10:**
* Walaupun menghasilkan akurasi sempurna (**1.0** atau **100%**), model tersebut **tidak valid secara operasional** karena mengalami *data leakage* (kebocoran informasi masa depan). Pada saat pihak akademik ingin memprediksi mahasiswa yang baru duduk di **semester 4**, mahasiswa tersebut belum menempuh skripsi sehingga data `nilai_ta` **sama sekali belum tersedia**. Oleh sebab itu, model ini tidak bisa diterapkan untuk prediksi dini di semester 4.

## Ringkasan untuk Lembar Kerja (Kelompok 10)

| Langkah | Tindakan Kelompok 10 |
|---|---|
| **Cleaning** | Menghapus 12 baris duplikat (`drop_duplicates()` menjadi 300 baris); menyamakan format teks `jenis_kelamin` (`L`/`P`) dan kapitalisasi `prodi` (`str.title()`); mengoreksi salah ketik `ipk_sem4 > 4` dengan membagi 10; mengubah angka anomali (`kehadiran > 100` dan `jam_belajar > 84`) menjadi `NaN`; lalu mengisi nilai kosong memakai `SimpleImputer` dengan `median` untuk kolom numerik dan `most_frequent` (modus) untuk kolom kategorikal. |
| **Scaling** | Menggunakan `StandardScaler()` pada kelima fitur numerik (`ipk_sem4`, `sks_lulus`, `kehadiran`, `jam_belajar`, `penghasilan_ortu`) di dalam `Pipeline` supaya seluruh variabel numerik terstandarisasi (rata-rata 0, standar deviasi 1) dan tidak didominasi oleh skala jutaan rupiah pada `penghasilan_ortu`. |
| **Encoding** | Menggunakan `OneHotEncoder()` untuk mengubah 4 fitur nominal (`jenis_kelamin`, `prodi`, `jalur_masuk`, `ikut_organisasi`) menjadi vektor biner, serta `OrdinalEncoder(categories=[["Dasar", "Menengah", "Mahir"]])` untuk mengubah fitur bertingkat `kemampuan_bahasa` menjadi urutan angka `0, 1, 2`. |
| **Feature yang dibuang** | Membuang 3 kolom: `nim` (atribut identitas unik tanpa nilai prediktif), `kampus` (atribut konstan bernilai `"Tembalang"` pada seluruh baris / varians nol), dan `nilai_ta` (atribut pemicu *data leakage* karena nilainya baru ada di semester akhir, bukan di semester 4). |
| **Pembagian data** | Melakukan *split* data sebesar 80% *train* (240 baris) dan 20% *test* (60 baris) menggunakan `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` **sebelum** tahapan imputasi, *scaling*, dan *encoding* agar rasio kelas tetap seimbang dan mencegah kebocoran statistik data uji ke proses pelatihan. |
| **Akurasi sebelum → sesudah** | Dari akurasi awal **0.619 (61,9%)** meningkat menjadi **0.833 (83,3%)** pada model KNN (**naik +0.214 / +21,4%**) serta **0.800 (80,0%)** pada model Regresi Logistik (**naik +0.181 / +18,1%**). *(Pada eksperimen tambahan, KNN tanpa scaling turun ke **0.617**, sedangkan penyertaan `nilai_ta` menghasilkan akurasi semu **1.000** akibat data leakage)*. |
