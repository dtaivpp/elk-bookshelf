import sycamore
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder

os_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }


index_settings = {
    "body": {
        "settings": {
            "index.knn": True,
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {"name": "hnsw", "engine": "faiss"},
                },
            },
        },
    },
}


paths = []
from os import walk, path
for root, dirs, files in walk("/Volumes/HomeMedia/media/PrintMedia/Educational/SoftwareEngineering"):
    for file in files:
        if file.endswith(".pdf"):
            paths.append(path.join(root, file))

context = sycamore.init()
pdf_docset = context.read.binary(paths, binary_format="pdf") \
                .partition(partitioner=UnstructuredPdfPartitioner()) \
                .explode() \
                .sketch() \
                .embed(SentenceTransformerEmbedder(
                        batch_size=10000, model_batch_size=1000,
                        model_name="sentence-transformers/all-MiniLM-L6-v2"))

pdf_docset.write.opensearch(
     os_client_args=os_client_args,
     index_name="test_sycamore",
     index_settings=index_settings)