# .env dosyasındaki gizli bilgileri (API anahtarları gibi) ortam değişkeni olarak yükler.
from dotenv import load_dotenv

# LangChain Hub üzerinden hazır prompt şablonları çekmek için kullanılır.
from langchain import hub

# Chroma adında bir vektör veritabanını kullanmak için bu sınıfı dahil ediyoruz.
from langchain_chroma import Chroma

# Web sitelerinden veri çekmek için kullanılan bir belge yükleyici (scraper).
from langchain_community.document_loaders import WebBaseLoader

# LLM (Large Language Model) cevabını düzgün bir metin olarak almak için output parser.
from langchain_core.output_parsers import StrOutputParser#aldığımız ürünlertin hepsini en son output_parser a vereceğiz en son

# Prompt oluştururken, kullanıcıdan gelen veriyi direkt iletmek için bir geçiş (passthrough) aracı.
from langchain_core.runnables import RunnablePassthrough #placeholder gibi kullanmak için promtun içerisine bunu kullanabiliyorz

# Metinleri embedding (sayısal vektör temsili) formatına çevirmek için OpenAI modelini kullanır.
from langchain_openai import OpenAIEmbeddings #indirdiklerimizi Vektörüze etmede

# Metni bölmek (chunk'lara ayırmak) için kullanılan akıllı bir bölücü.
# Belirli uzunlukta parçalar üretir ve bölümler arasında biraz örtüşme sağlar.
from langchain_text_splitters import RecursiveCharacterTextSplitter#internetten indirdiğimiz verileri adım adım bölmemize batch batch chroma database e kaydetmemize yarayacak

# OpenAI'nin Chat LLM (chat modeli) sınıfı. Buradan ChatGPT gibi modellerle çalışabiliyoruz.
from langchain_openai import ChatOpenAI #openAI a istek atabilmek için

# Web sayfasını parse ederken belirli HTML elementlerini filtrelemek için BeautifulSoup'un parçası.
import bs4

load_dotenv()# Ortam değişkenlerini (.env dosyasındaki) sisteme yükler.

# GPT-3.5-turbo-0125 modelini kullanarak bir ChatOpenAI nesnesi oluşturduk.
# Bu nesne, LLM'e istek atmak ve cevap almak için kullanılacak.
llm = ChatOpenAI(model="gpt-3.5-turbo-0125") #LLM imizi olştrdk



# 1. AŞAMA: VERİYİ WEB’TEN ÇEKME
# Web'den veri çekecek olan WebBaseLoader sınıfının örneğini oluşturduk.
# Bu loader, belirttiğimiz URL'den veri alacak ve sadece belirttiğimiz HTML sınıflarını işleyecek.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")# Yalnızca bu sınıflardaki içeriği al. WEB sitesinnden web sitesaine değişebilir.
        )
    ),
)
# Yukarıdaki loader aracılığıyla URL'deki verileri çektik.
# docs değişkeni artık bu sayfadan çekilen ham verileri içeriyor (HTML filtrelemesinden geçirilmiş).
docs = loader.load()


#2. AŞAMA: METNİ PARÇALAMA
# Veriyi 1000 karakterlik bloklara ayırıyoruz, her blok arasında 200 karakterlik örtüşme olacak.
# Bu örtüşme, bağlamın korunmasına yardımcı olur.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Yüklenen dokümanları parçalara ayırıyoruz.
splits = text_splitter.split_documents(docs)



# 3. AŞAMA: VEKTÖR VERİ TABANI (Chroma)
# Her bir metin parçasını embedding (vektör) formatına çevirerek Chroma veri tabanına kaydediyoruz.
# Bu embedding işlemi, metinleri anlam olarak karşılaştırabilmek için gerekli.
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())


# 4. AŞAMA: VERİ GETİRİCİ (Retriever)
# Vektör veritabanından sorguya en uygun parçaları döndürecek retriever oluşturuluyor.
retriever = vectorstore.as_retriever()


# 5. AŞAMA: HAZIR PROMPT ŞABLONU
# LangChain Hub'dan hazır bir RAG prompt şablonu çekiyoruz.
# Bu prompt, context (bağlam) ve question (soru) alanlarını kullanarak LLM'e yönlendirilecek.
prompt = hub.pull("rlm/rag-prompt")


#6. AŞAMA: FORMATLAYICI YARDIMCI FONKSİYON
# Retriever'dan gelen dokümanların içeriğini tek bir metin haline getiriyoruz.
# LLM'e context olarak verilecek.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#7. AŞAMA: RAG ZİNCİRİNİ OLUŞTURMA
# Retrieval-Augmented Generation (RAG) zinciri tanımlanıyor.

# 1. {"context": retriever | format_docs, "question": RunnablePassthrough()}
#    -> Soru prompta gidecek, context kısmı ise retriever'dan gelecek belgelerin işlenmiş hali olacak.

# 2. | prompt
#    -> Yukarıdaki yapıdan elde edilen sözlük (question + context), prompt şablonuna aktarılıyor.

# 3. | llm
#    -> Prompt, LLM'e gönderiliyor.

# 4. | StrOutputParser()
#    -> LLM'den gelen cevabı sadeleştirip, düzgün bir çıktı olarak veriyor.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# 8. AŞAMA: SORU SORUP CEVAP ALMA
if __name__ == "__main__":
    for chunk in rag_chain.stream("What is Task Decomposition?"):
        # rag_chain.stream fonksiyonu, cevabı adım adım (stream halinde) verir.
        # "What is Task Decomposition?" sorusunu modelimize yöneltiyoruz.
        print(chunk, end="", flush=True)#flash=true ile öncekileri tutmadan cevaplamış olacak
        #3 cümle ile cevap verdi