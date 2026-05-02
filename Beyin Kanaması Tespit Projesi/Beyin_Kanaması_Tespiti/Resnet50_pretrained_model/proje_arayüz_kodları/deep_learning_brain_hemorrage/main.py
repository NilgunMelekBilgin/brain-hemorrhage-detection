import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys

# --- Kütüphane kontrolleri ---
try:
    import numpy as np
except ImportError:
    print("HATA: numpy kurulu değil. Çalıştırın: pip install numpy")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("HATA: Pillow kurulu değil. Çalıştırın: pip install pillow")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("UYARI: TensorFlow bulunamadı. Demo modu aktif.")

# ─────────────────────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/best_resnet50_model.keras"   # model dosyanızın adı
IMG_SIZE      = (224, 224)          # ResNet50 giriş boyutu
THRESHOLD     = 0.5                 # karar eşiği

RENK_BG       = "#0f1117"
RENK_PANEL    = "#1a1d27"
RENK_KART     = "#222536"
RENK_BORDER   = "#2e3250"
RENK_MAVI     = "#4a90d9"
RENK_YESIL    = "#2ecc71"
RENK_KIRMIZI  = "#e74c3c"
RENK_SARI     = "#f39c12"
RENK_METIN    = "#e8eaf6"
RENK_METIN2   = "#8b90b0"
RENK_BEYAZ    = "#ffffff"

FONT_BASLIK   = ("Segoe UI", 14, "bold")
FONT_ALT      = ("Segoe UI", 10)
FONT_KUCUK    = ("Segoe UI", 9)
FONT_SONUC    = ("Segoe UI", 22, "bold")
FONT_MONO     = ("Consolas", 10)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────
model = None

def modeli_yukle():
    global model
    if not TF_AVAILABLE:
        return False
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        model = load_model(MODEL_PATH)
        return True
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return False

def goruntu_isle(img_path):
    """Görüntüyü yükle, ön işle ve modele ver."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def tahmin_yap(img_path):
    """
    Gerçek model varsa: modeli kullan.
    Model yoksa: demo rastgele tahmin döndür.
    """
    if model is not None and TF_AVAILABLE:
        arr = goruntu_isle(img_path)
        prob = float(model.predict(arr, verbose=0)[0][0])
    else:
        # DEMO MODU — model yokken test için
        import random
        prob = random.uniform(0.1, 0.95)

    sinif = 1 if prob >= THRESHOLD else 0
    return prob, sinif

# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI — Renk çubuğu (canvas canvas'ta)
# ─────────────────────────────────────────────────────────────────────────────
def prob_rengi(prob):
    if prob >= 0.75:
        return RENK_KIRMIZI
    elif prob >= 0.50:
        return RENK_SARI
    else:
        return RENK_YESIL

# ─────────────────────────────────────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────
class BeynApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Beyin Kanaması Tespit Sistemi")
        self.root.geometry("900x680")
        self.root.minsize(780, 580)
        self.root.configure(bg=RENK_BG)

        self.goruntu_yolu   = None
        self.model_yuklu    = False
        self.analiz_calisiyor = False

        self._arayuz_kur()
        self._model_yukle_arkaplanda()

    # ── UI kurulum ─────────────────────────────────────────────────────────
    def _arayuz_kur(self):
        # Başlık
        baslik_frame = tk.Frame(self.root, bg=RENK_BG)
        baslik_frame.pack(fill="x", padx=24, pady=(20, 0))

        tk.Label(
            baslik_frame,
            text="Beyin Kanaması Tespit Sistemi",
            font=("Segoe UI", 16, "bold"),
            bg=RENK_BG, fg=RENK_METIN
        ).pack(side="left")

        self.model_durumu = tk.Label(
            baslik_frame,
            text="● Model yükleniyor...",
            font=FONT_KUCUK,
            bg=RENK_BG, fg=RENK_SARI
        )
        self.model_durumu.pack(side="right", pady=4)

        tk.Label(
            self.root,
            text="ResNet50 Transfer Learning  ·  %96.7 Doğruluk  ·  F1: 0.966",
            font=FONT_KUCUK,
            bg=RENK_BG, fg=RENK_METIN2
        ).pack(anchor="w", padx=24)

        ayirici = tk.Frame(self.root, bg=RENK_BORDER, height=1)
        ayirici.pack(fill="x", padx=24, pady=12)

        # Ana içerik: sol (görüntü) + sağ (sonuç)
        icerik = tk.Frame(self.root, bg=RENK_BG)
        icerik.pack(fill="both", expand=True, padx=24, pady=0)

        # Sol panel
        sol = tk.Frame(icerik, bg=RENK_PANEL, bd=0, highlightthickness=1,
                       highlightbackground=RENK_BORDER)
        sol.pack(side="left", fill="both", expand=True, padx=(0, 8))

        tk.Label(sol, text="BT Görüntüsü", font=FONT_BASLIK,
                 bg=RENK_PANEL, fg=RENK_METIN).pack(anchor="w", padx=16, pady=(14, 0))
        tk.Label(sol, text="PNG, JPG formatında BT taraması yükleyin",
                 font=FONT_KUCUK, bg=RENK_PANEL, fg=RENK_METIN2).pack(anchor="w", padx=16)

        # Görüntü alanı
        self.goruntu_cerceve = tk.Frame(sol, bg="#111420", bd=0,
                                        highlightthickness=1, highlightbackground=RENK_BORDER)
        self.goruntu_cerceve.pack(fill="both", expand=True, padx=16, pady=12)

        self.goruntu_label = tk.Label(
            self.goruntu_cerceve,
            text="Görüntü yüklemek için\naşağıdaki butona tıklayın",
            font=FONT_ALT, bg="#111420", fg=RENK_METIN2,
            justify="center"
        )
        self.goruntu_label.pack(expand=True)

        # Dosya adı
        self.dosya_label = tk.Label(sol, text="", font=FONT_KUCUK,
                                    bg=RENK_PANEL, fg=RENK_METIN2)
        self.dosya_label.pack(pady=(0, 4))

        # Butonlar
        btn_frame = tk.Frame(sol, bg=RENK_PANEL)
        btn_frame.pack(fill="x", padx=16, pady=(0, 14))

        self.yukle_btn = tk.Button(
            btn_frame, text="Görüntü Seç",
            font=FONT_ALT, bg=RENK_KART, fg=RENK_METIN,
            activebackground=RENK_BORDER, activeforeground=RENK_BEYAZ,
            relief="flat", bd=0, padx=14, pady=8, cursor="hand2",
            command=self.goruntu_sec
        )
        self.yukle_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.analiz_btn = tk.Button(
            btn_frame, text="Analiz Başlat",
            font=("Segoe UI", 10, "bold"), bg=RENK_MAVI, fg=RENK_BEYAZ,
            activebackground="#357abd", activeforeground=RENK_BEYAZ,
            relief="flat", bd=0, padx=14, pady=8, cursor="hand2",
            state="disabled", command=self.analiz_baslat
        )
        self.analiz_btn.pack(side="left", expand=True, fill="x")

        # Sağ panel
        sag = tk.Frame(icerik, bg=RENK_PANEL, width=300, bd=0,
                       highlightthickness=1, highlightbackground=RENK_BORDER)
        sag.pack(side="right", fill="both", padx=(8, 0))
        sag.pack_propagate(False)

        tk.Label(sag, text="Analiz Sonucu", font=FONT_BASLIK,
                 bg=RENK_PANEL, fg=RENK_METIN).pack(anchor="w", padx=16, pady=(14, 0))

        # Sonuç kutusu
        self.sonuc_cerceve = tk.Frame(sag, bg=RENK_KART, bd=0,
                                      highlightthickness=1, highlightbackground=RENK_BORDER)
        self.sonuc_cerceve.pack(fill="x", padx=16, pady=10)

        self.sonuc_baslik = tk.Label(
            self.sonuc_cerceve, text="Bekleniyor",
            font=FONT_SONUC, bg=RENK_KART, fg=RENK_METIN2
        )
        self.sonuc_baslik.pack(pady=(16, 4))

        self.sonuc_aciklama = tk.Label(
            self.sonuc_cerceve,
            text="Görüntü yüklenip analiz\nbaşlatıldığında sonuç\ngörüntülenir",
            font=FONT_ALT, bg=RENK_KART, fg=RENK_METIN2,
            justify="center"
        )
        self.sonuc_aciklama.pack(pady=(0, 16))

        # Olasılık çubuğu
        tk.Label(sag, text="Kanama Olasılığı", font=FONT_KUCUK,
                 bg=RENK_PANEL, fg=RENK_METIN2).pack(anchor="w", padx=16)

        self.prob_canvas = tk.Canvas(sag, height=22, bg=RENK_PANEL,
                                     highlightthickness=0)
        self.prob_canvas.pack(fill="x", padx=16, pady=(4, 0))
        self.prob_canvas.bind("<Configure>", self._prob_cubugu_ciz)

        self.prob_yuzde = tk.Label(sag, text="— %", font=("Segoe UI", 13, "bold"),
                                   bg=RENK_PANEL, fg=RENK_METIN)
        self.prob_yuzde.pack(anchor="w", padx=16, pady=(4, 10))

        # İstatistik kartları
        istat_frame = tk.Frame(sag, bg=RENK_PANEL)
        istat_frame.pack(fill="x", padx=16)
        istat_frame.columnconfigure(0, weight=1)
        istat_frame.columnconfigure(1, weight=1)

        self.istat_etiketler = {}
        for i, (anahtar, etiket, varsayilan) in enumerate([
            ("sinif",   "Sınıf",       "—"),
            ("guven",   "Güven",       "—"),
            ("model",   "Model",       "ResNet50"),
            ("dogruluk","Doğruluk",    "%96.7"),
        ]):
            satir, sutun = divmod(i, 2)
            kart = tk.Frame(istat_frame, bg=RENK_KART, bd=0,
                            highlightthickness=1, highlightbackground=RENK_BORDER)
            kart.grid(row=satir, column=sutun, padx=(0 if sutun else 0, 4 if sutun == 0 else 0),
                      pady=4, sticky="ew", ipadx=10, ipady=6)
            tk.Label(kart, text=etiket, font=FONT_KUCUK,
                     bg=RENK_KART, fg=RENK_METIN2).pack(anchor="w", padx=10, pady=(6, 0))
            lbl = tk.Label(kart, text=varsayilan, font=("Segoe UI", 10, "bold"),
                           bg=RENK_KART, fg=RENK_METIN)
            lbl.pack(anchor="w", padx=10, pady=(0, 6))
            self.istat_etiketler[anahtar] = lbl

        # Yükleme göstergesi
        self.yukleme_label = tk.Label(sag, text="", font=FONT_KUCUK,
                                      bg=RENK_PANEL, fg=RENK_MAVI)
        self.yukleme_label.pack(pady=8)

        # Uyarı notu
        ayirici2 = tk.Frame(self.root, bg=RENK_BORDER, height=1)
        ayirici2.pack(fill="x", padx=24, pady=(12, 0))

        tk.Label(
            self.root,
            text="⚠  Bu araç yalnızca araştırma amaçlıdır. Klinik tanı için kullanılamaz."
                 "  Gerçek değerlendirme için uzman hekim görüşü alınız.",
            font=("Segoe UI", 8),
            bg=RENK_BG, fg="#5a5f7a", wraplength=850, justify="center"
        ).pack(pady=8)

        self._prob_deger = 0.0

    # ── Olasılık çubuğu ────────────────────────────────────────────────────
    def _prob_cubugu_ciz(self, event=None):
        c = self.prob_canvas
        c.delete("all")
        w = c.winfo_width()
        if w < 2:
            return
        h = 18
        # Arka plan
        c.create_rectangle(0, 2, w, h, fill=RENK_KART, outline="")
        # Dolu kısım
        dolu = int(w * self._prob_deger)
        if dolu > 0:
            renk = prob_rengi(self._prob_deger)
            c.create_rectangle(0, 2, dolu, h, fill=renk, outline="")

    def _prob_guncelle(self, prob):
        self._prob_deger = prob
        self._prob_cubugu_ciz()
        self.prob_yuzde.config(
            text=f"{round(prob * 100)} %",
            fg=prob_rengi(prob)
        )

    # ── Model yükleme ───────────────────────────────────────────────────────
    def _model_yukle_arkaplanda(self):
        def yukle():
            basarili = modeli_yukle()
            self.root.after(0, lambda: self._model_durumu_guncelle(basarili))
        threading.Thread(target=yukle, daemon=True).start()

    def _model_durumu_guncelle(self, basarili):
        if basarili:
            self.model_yuklu = True
            self.model_durumu.config(text="● Model hazır", fg=RENK_YESIL)
        elif not TF_AVAILABLE:
            self.model_durumu.config(text="● Demo modu (TF yok)", fg=RENK_SARI)
        else:
            self.model_durumu.config(
                text=f"● Model bulunamadı ({MODEL_PATH})", fg=RENK_KIRMIZI
            )

    # ── Görüntü seçme ───────────────────────────────────────────────────────
    def goruntu_sec(self):
        yol = filedialog.askopenfilename(
            title="BT Görüntüsü Seç",
            filetypes=[
                ("Resim Dosyaları", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        if not yol:
            return

        self.goruntu_yolu = yol
        self._goruntu_goster(yol)
        dosya_adi = os.path.basename(yol)
        boyut_kb  = os.path.getsize(yol) / 1024
        self.dosya_label.config(text=f"{dosya_adi}  ·  {boyut_kb:.1f} KB")
        self.analiz_btn.config(state="normal")
        self._sonuc_sifirla()

    def _goruntu_goster(self, yol):
        try:
            img = Image.open(yol)
            img.thumbnail((340, 300), Image.LANCZOS)
            foto = ImageTk.PhotoImage(img)
            self.goruntu_label.config(image=foto, text="")
            self.goruntu_label._foto = foto   # referans tut
        except Exception as e:
            self.goruntu_label.config(
                text=f"Görüntü açılamadı:\n{e}", image=""
            )

    # ── Sonuç sıfırla ───────────────────────────────────────────────────────
    def _sonuc_sifirla(self):
        self.sonuc_baslik.config(text="Hazır", fg=RENK_METIN2)
        self.sonuc_aciklama.config(text="Analiz başlatmak için\n'Analiz Başlat' butonuna tıklayın")
        self.sonuc_cerceve.config(highlightbackground=RENK_BORDER)
        self._prob_guncelle(0)
        self.prob_yuzde.config(text="— %", fg=RENK_METIN)
        self.istat_etiketler["sinif"].config(text="—", fg=RENK_METIN)
        self.istat_etiketler["guven"].config(text="—", fg=RENK_METIN)

    # ── Analiz başlat ───────────────────────────────────────────────────────
    def analiz_baslat(self):
        if self.analiz_calisiyor or not self.goruntu_yolu:
            return

        self.analiz_calisiyor = True
        self.analiz_btn.config(state="disabled", text="Analiz ediliyor...")
        self.yukleme_label.config(text="Görüntü işleniyor...")

        def calis():
            try:
                prob, sinif = tahmin_yap(self.goruntu_yolu)
                self.root.after(0, lambda: self._sonuc_goster(prob, sinif))
            except Exception as e:
                self.root.after(0, lambda: self._hata_goster(str(e)))

        threading.Thread(target=calis, daemon=True).start()

    def _sonuc_goster(self, prob, sinif):
        self.analiz_calisiyor = False
        self.analiz_btn.config(state="normal", text="Analiz Başlat")
        self.yukleme_label.config(text="")

        pct = round(prob * 100)

        if sinif == 1:
            metin   = "⚠  Kanama Tespit Edildi"
            renk    = RENK_KIRMIZI
            aciklama = (f"Model bu görüntüde beyin kanaması\nolasılığını %{pct} olarak\ntespit etti."
                        if prob >= 0.75 else
                        f"Orta düzeyde kanama şüphesi.\nOlasılık: %{pct}.")
            guven = "Yüksek" if prob >= 0.75 else "Orta"
        else:
            metin   = "✓  Kanama Tespit Edilmedi"
            renk    = RENK_YESIL
            aciklama = f"Model bu görüntüde anormal\nbulgu tespit etmedi.\nOlasılık: %{pct}."
            guven = "Yüksek" if prob <= 0.25 else "Orta"

        self.sonuc_baslik.config(text=metin, fg=renk)
        self.sonuc_aciklama.config(text=aciklama, fg=RENK_METIN)
        self.sonuc_cerceve.config(highlightbackground=renk)
        self._prob_guncelle(prob)
        self.istat_etiketler["sinif"].config(
            text="Kanama Var" if sinif == 1 else "Kanama Yok",
            fg=renk
        )
        self.istat_etiketler["guven"].config(text=guven, fg=RENK_METIN)

    def _hata_goster(self, mesaj):
        self.analiz_calisiyor = False
        self.analiz_btn.config(state="normal", text="Analiz Başlat")
        self.yukleme_label.config(text="")
        messagebox.showerror("Hata", f"Analiz sırasında hata oluştu:\n\n{mesaj}")


# ─────────────────────────────────────────────────────────────────────────────
# BAŞLAT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    uygulama = BeynApp(root)
    root.mainloop()