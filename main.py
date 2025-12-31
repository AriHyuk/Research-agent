import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI(title="API Riset Jurnal (Sequential + Grounding)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Artinya: "Semua website boleh akses API ini"
    allow_credentials=True,
    allow_methods=["*"],  # Boleh GET, POST, PUT, DELETE, dll
    allow_headers=["*"],  # Boleh kirim header apa aja
)
# --- SETUP AI ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "hopeful-flame-464506-k6")
LOCATION = "global"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- FUNGSI PEMBANTU (Manggil AI) ---
def panggil_gemini(model, prompt, tools=None):
    config = types.GenerateContentConfig(
        temperature=0.3, # Rendah biar serius
        tools=tools
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    return response

# --- MODEL REQUEST ---
class JurnalRequest(BaseModel):
    topik: str
    target_pembaca: str = "Dosen Penguji"

@app.get("/")
def root():
    return {"status": "Siap", "mode": "Sequential Agent"}

@app.post("/api/riset-lengkap")
def sequential_research(data: JurnalRequest):
    print(f"🚀 Memulai Misi Sequential untuk: {data.topik}")
    
    try:
        # ---------------------------------------------------------
        # STEP 1: RESEARCHER AGENT (Mata-mata Google)
        # ---------------------------------------------------------
        print("🕵️ Step 1: Researcher sedang Googling...")
        prompt_researcher = f"""
        PERAN: Academic Researcher.
        TOPIK: {data.topik}
        TUGAS: 
        1. Cari fakta, data statistik terbaru, dan teori relevan menggunakan Google Search.
        2. Kumpulkan poin-poin kuncinya saja (Raw Data).
        3. JANGAN mengarang, wajib dari hasil search.
        """
        
        # Aktifkan Google Search Tool CUMA di tahap ini
        tools_grounding = [types.Tool(google_search=types.GoogleSearch())]
        
        hasil_step_1 = panggil_gemini('gemini-2.5-flash', prompt_researcher, tools=tools_grounding)
        
        # Ambil Metadata Sumber (Biar keren ada buktinya)
        sumber_validasi = []
        if hasil_step_1.candidates[0].grounding_metadata.search_entry_point:
            sumber_validasi.append("Data Terverifikasi Google Search")

        # ---------------------------------------------------------
        # STEP 2: WRITER AGENT (Penulis Draft)
        # ---------------------------------------------------------
        print("✍️ Step 2: Writer sedang menulis...")
        prompt_writer = f"""
        PERAN: Academic Writer.
        DATA MENTAH DARI RESEARCHER: 
        {hasil_step_1.text}

        TUGAS:
        Ubah data mentah di atas menjadi Narasi Jurnal Ilmiah yang mengalir.
        Target Pembaca: {data.target_pembaca}.
        Struktur: Pendahuluan -> Pembahasan Utama -> Kesimpulan.
        """
        
        # Tidak perlu tools search, cukup olah teks
        hasil_step_2 = panggil_gemini('gemini-2.5-flash', prompt_writer)

        # ---------------------------------------------------------
        # STEP 3: EDITOR AGENT (Si Anti Halu / Validasi)
        # ---------------------------------------------------------
        print("🧐 Step 3: Editor sedang review...")
        prompt_editor = f"""
        PERAN: Senior Editor (Quality Control).
        DRAFT TULISAN:
        {hasil_step_2.text}

        TUGAS:
        1. Cek apakah ada kalimat yang terdengar tidak logis? Hapus.
        2. Perbaiki gaya bahasa agar sangat formal dan akademis.
        3. Pastikan alurnya enak dibaca.
        4. Outputkan HANYA hasil perbaikan finalnya.
        """
        
        # Pakai model yang agak pinteran dikit kalau ada (atau flash juga oke)
        hasil_step_3 = panggil_gemini('gemini-2.5-flash', prompt_editor)
        google_widget_html = ""
        try:
            # Cek apakah Step 1 punya metadata grounding
            if hasattr(hasil_step_1, 'candidates') and hasil_step_1.candidates:
                cand = hasil_step_1.candidates[0]
                if hasattr(cand, 'grounding_metadata') and cand.grounding_metadata.search_entry_point:
                    google_widget_html = cand.grounding_metadata.search_entry_point.rendered_content
        except Exception as e:
            print(f"Gagal ambil widget grounding: {e}")
        # ---------------------------------------------------------
        # FINAL OUTPUT
        # ---------------------------------------------------------
        return {
            "status": "success",
            "tahapan": {
                "1_riset_raw": hasil_step_1.text[:200] + "...", # Preview dikit
                "2_draft_awal": hasil_step_2.text[:200] + "..."
            },
            "hasil_final": hasil_step_3.text, # Ini yang dipake di web kamu
            "sumber_html": google_widget_html
            
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))