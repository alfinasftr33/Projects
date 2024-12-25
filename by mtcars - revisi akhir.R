library(psych)
library(ggplot2)
library(GPArotation)
library(lavaan)

# Import data
data(mtcars)
View(mtcars)

# menampilkan beberapa baris pertama
head(mtcars)

# ukuran data 
dim(mtcars)

# struktur data
str(mtcars)

# daftar nama variabel
colnames(mtcars)

# Preprocessing dan Eksplorasi Data
# ----------------------------------

# Cek data awal
summary(mtcars)

# Menghitung jumlah missing values per variabel
sum(is.na(mtcars))

# Mengecek persentase missing values per variabel
colMeans(is.na(mtcars)) * 100

# Membuat boxplot untuk setiap variabel
boxplot(mtcars)

# Mengecek nilai-nilai outlier per variabel
outliers <- boxplot(mtcars)$out

# Menggunakan uji Shapiro-Wilk untuk setiap variabel dalam dataset mtcars
for (column in names(mtcars)) {p_value <- shapiro.test(mtcars[[column]])$p.value
# Menampilkan hasil
  cat("Variabel:", column, "\tP-value:", p_value, "\n")}

# Menggunakan uji normalitas Shapiro-Wilk pada variabel mpg
shapiro.test(mtcars$mpg)

# Membuat plot QQ untuk setiap variabel
qqnorm(mtcars$mpg)
qqline(mtcars$mpg)

# Standarisasi variabel-variabel dalam dataset
standardized_mtcars <- scale(mtcars)
standardized_mtcars

# Analisis Faktor
# -----------------
# Matriks korelasi (hanya untuk mengecek)
cor_matrix <- cor(mtcars)
cor_matrix
dim(cor_matrix)

# sebelum masuk ke uji selanjutnya, bisa dicantumkan tujuan dari studi kasusnya dulu
# Uji Bartlett
bartlett_test <- psych::cortest.bartlett(cor_matrix, n = nrow(mtcars))
bartlett_test

# Uji Kesesuaian Sampling Kaiser-Meyer-Olkin (KMO)
kmo_test <- psych::KMO(cor_matrix)
kmo_test

# Analisis Faktor Eksploratori (EFA) dengan metode PCA
efa_result <- psych::principal(mtcars, nfactors = 3, rotate = "varimax")
summary(efa_result)

# menentukan jumlah faktor dengan Scree Plot
# -----------------------
# Menghitung nilai eigen
ev <- efa_result$values
ev
# Menentukan jumlah faktor berdasarkan kriteria nilai eigen
num_factors <- sum(ev > 1)  # Misalnya, menggunakan kriteria eigen > 1
print(num_factors)

# Melakukan uji paralel
parallel_analysis <- fa.parallel(mtcars, fm = "pc", fa = "fa", n.iter = 100, main = "Parallel Analysis")

# Menampilkan scree plot (ini di aku gak muncul hasilnya)
plot(parallel_analysis$eigen$values, type = "b", xlab = "Number of Factors", ylab = "Eigenvalues", main = "Scree Plot")

# Membuat data frame untuk scree plot (bisanya pakai ini)
df <- data.frame(n = 1:length(ev), ev)

# Scatter plot
ggplot(df, aes(x = n, y = ev)) +
  geom_point() +
  labs(title = "Scree Plot", x = "Factors", y = "Eigen Value") +
  theme_minimal()

# Line plot
ggplot(df, aes(x = n, y = ev)) +
  geom_line() +
  labs(title = "Scree Plot", x = "Factors", y = "Eigen Value") +
  theme_minimal()

# Melakukan analisis faktor menggunakan dataset mtcars
factor_analysis <- fa(mtcars, nfactors = length(mtcars), rotate = "varimax")
factor_analysis # bisa juga untuk menampilkan seluruh faktor sebelum ekstrasi 

# Menampilkan scree plot (ini di aku gak muncul juga)
screeplot(factor_analysis, main = "Scree Plot")

# untuk dua faktor (ekstrasi faktor/seleksi faktor yang dibentuk)
efa_result <- principal(mtcars, nfactors = 2, rotate = "varimax")
efa_result # menampilkan faktor yang diambil 
fa.diagram(efa_result, main = "data")

# untuk tiga faktor (ekstrasi faktor/seleksi faktor yang dibentuk)
# untuk yang tiga faktor, ditunjukan hanya untuk membandingkan nilai dari dua faktor saja 
efa_result1 <- principal(mtcars, nfactors = 3, rotate = "varimax")
efa_result1 
fa.diagram(efa_result1, main = "data")

# rotasi Faktor dengan menggunakan varimax
# sebelum melakukan rotasi faktor, dijelaskan/interpretasi loadings pd masing2 x dan kenapa harus melakukan rotasi
# ---------------------------------------
efa_result
rotated_efa_result <- varimax(efa_result$loadings)
rotated_efa_result

# visualisasi hasil matirks loading dari hasil rotasi faktor
# Mengubah matriks loadings menjadi data frame
loadings_df <- as.data.frame.matrix(rotated_efa_result$loadings)
loadings_df

# Menampilkan plot bar untuk matriks loading
# bersamaan dengan menjelaskan makna loading pd masing2 variabel
barplot(t(loadings_df), beside = TRUE, col = rainbow(ncol(loadings_df)),
        main = "Matriks Loading", xlab = "Variabel", ylab = "Loading",
        legend.text = colnames(loadings_df), args.legend = list(x = "topright"))
abline(h = 0, lty = 2)

# interpretasi faktor
# --------------------
# Memperoleh loadings dari hasil analisis faktor sebelumnya (setelah dirotated)
# ini hanya untuk recall
loadings <- rotated_efa_result$loadings
print(loadings)

# Menampilkan faktor yang memiliki loadings tertinggi untuk setiap variabel
max_loadings <- apply(loadings, 2, function(x) names(x)[which.max(abs(x))])
print(max_loadings)

# Menampilkan faktor dengan loadings terbesar untuk setiap observasi
dominant_factor <- colnames(loadings)[apply(loadings, 1, which.max)]
print(dominant_factor)

# Menampilkan kontribusi masing-masing variabel terhadap faktor
# Nilai contribution biasanya dihitung dengan mengkuadratkan nilai loadings dan kemudian dinormalisasi dengan membaginya dengan jumlah total kontribusi semua variabel dalam faktor
contribution_values <- (loadings_df^2) / sum(loadings_df^2)
contribution_values

# ----------------------------------------------------------------------------------------------------
# verifikasi validasi (gak usah dicantumin di makalah)
# ------------------
# Melakukan verifikasi reliabilitas faktor dengan pengecekan dan pembalikan otomatis
reliability <- psych::alpha(rotated_efa_result$loadings, check.keys = TRUE)
print(reliability$total$raw_alpha)

# Melakukan uji konfirmatori (CFA) menggunakan paket sem
model <- 'f1 =~ carb + qsec + am'

# Membuat model fit
fit <- lavaan::cfa(model, data = mtcars)

# Menampilkan hasil uji CFA
summary(fit)
# Menggunakan fungsi fitMeasures()
fit_measures <- fitMeasures(fit, c("rmsea", "cfi", "tli"))

# Menggunakan fungsi cfit()
cfit <- cfit(fit, fit.measures = TRUE)

# Menampilkan metrik goodness of fit
print(fit_measures)
print(cfit)

# Menghitung validasi faktor
factor_validity <- lavaan::parameterEstimates(fit, standardized = TRUE)
print(factor_validity)





