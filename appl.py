import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from matplotlib import cm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns


documents = [
    # 1
    "Renaissance fashion in the 15th century was characterized by elaborate garments, rich fabrics, and detailed embroidery. Nobles wore doublets, hose, and ornate headdresses, symbolizing status and wealth.",
    # 2
    "Streetwear is heavily influenced by skate and surf culture, often featuring distressed jeans, graphic T-shirts, caps, sneakers, and hoodies for a casual aesthetic.",
    # 3
    "Art Nouveau fashion introduced flowing lines, floral motifs, and a focus on organic shapes, reflecting the broader artistic movement of the late 19th to early 20th century.",
    # 4
    "Urban style in modern times blends practicality with trends, emphasizing versatile items such as bomber jackets, joggers, and stylish backpacks.",
    # 5
    "Japanese avant-garde designers like Rei Kawakubo and Yohji Yamamoto redefined silhouettes with experimental cuts, challenging traditional definitions of shape and form.",
    # 6
    "Sportswear focuses on performance, breathability, and ease of movement. Tracksuits, running shoes, and sweat-wicking fabrics are key components of this category.",
    # 7
    "Victorian era fashion for women featured corsets, elaborate crinoline skirts, and high necklines, while men wore frock coats, waistcoats, and top hats.",
    # 8
    "Minimalism in clothing embraces simple lines, neutral colors, and high-quality fabrics, reflecting a more functional and understated approach to fashion.",
    # 9
    "Haute couture represents the pinnacle of fashion design, emphasizing handcrafted, one-of-a-kind garments tailored to individual clients, often showcased on exclusive runways in Paris.",
    # 10
    "Athleisure merges sportswear with everyday style, focusing on comfort, performance, and fashionable design elements like mesh details or metallic accents.",
    # 11
    "Flapper dresses of the 1920s were straight and loose, featuring dropped waists, beaded fringes, and shorter hemlines, symbolizing women's changing social roles.",
    # 12
    "Contemporary street style draws heavily from hip-hop culture, featuring oversized jackets, chunky sneakers, bold logos, and bright color palettes.",
    # 13
    "Eco-friendly or sustainable fashion is a rising trend, emphasizing organic materials, ethical labor practices, and reducing environmental impact.",
    # 14
    "In the 1960s, mod fashion popularized short hemlines, bold geometric patterns, and a youthful spirit. The shift dress became a symbol of modern femininity.",
    # 15
    "The punk movement of the 1970s brought distressed clothing, leather jackets, safety pins, and a rebellious ethos, challenging mainstream fashion norms."
]

# Шаг 1: Векторизация документов (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Шаг 2: Применение TruncatedSVD (LSA) для снижения размерности до 2D
lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X)

# Получаем список терминов, используемых при векторизации
terms = vectorizer.get_feature_names_out()

# Шаг 3: Анализ наиболее важных терминов для каждой «скрытой» компоненты
num_terms_to_show = 7  # выводим топ-7 терминов на каждую компоненту

print("Top terms per LSA component:")
for i, component in enumerate(lsa.components_):
    sorted_indices = np.argsort(component)[::-1]
    top_terms = [terms[idx] for idx in sorted_indices[:num_terms_to_show]]
    print(f"Component {i+1}: {top_terms}")
print()

# Генерируем цвета для каждого документа на основе колормэпа
num_docs = len(documents)
colors = cm.rainbow(np.linspace(0, 1, num_docs))

plt.figure(figsize=(10, 7))

for i, (x_coord, y_coord) in enumerate(X_lsa):
    plt.scatter(x_coord, y_coord, color=colors[i], s=50, alpha=0.7, label=f"Doc {i+1}")
    # Вывод текста рядом с точкой, чтобы легче идентифицировать документ
    plt.text(x_coord + 0.01, y_coord + 0.01, f"D{i+1}", fontsize=9)

plt.title("Документы в 2D-пространстве LSA (Одежда/Мода)")
plt.xlabel("LSA Component 1")
plt.ylabel("LSA Component 2")
plt.grid(True)

plt.show()


# 1. Извлечение эмбеддингов посредством модели SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# 2. Задаём число кластеров (например, 4) и обучаем модель K-means
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# 3. Снижение размерности (t-SNE) для визуализации на двумерном пространстве
tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=2000)
embeddings_2d = tsne.fit_transform(embeddings)

# 4. Отрисовка результатов
plt.figure(figsize=(10, 7))
palette = sns.color_palette("hls", num_clusters)

sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], hue=labels,
                palette=palette, s=100, alpha=0.8, legend='full')

for i, txt in enumerate(documents):
    plt.annotate(str(i+1), (embeddings_2d[i,0]+0.3, embeddings_2d[i,1]+0.3),
                 fontsize=9)

plt.title("K-Means кластеризация текстов Одежда/Мода (t-SNE visualization)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Cluster")
plt.show()

# Печать результатов кластеризации
for cluster_id in range(num_clusters):
    print(f"\nКластер {cluster_id}:")
    for i, label in enumerate(labels):
        if label == cluster_id:
            print(f" - Документ {i+1}: {documents[i]}")

#
# Кластер 0:
#  - Документ 6: Sportswear focuses on performance, breathability, and ease of movement. Tracksuits, running shoes, and sweat-wicking fabrics are key components of this category.
#  - Документ 10: Athleisure merges sportswear with everyday style, focusing on comfort, performance, and fashionable design elements like mesh details or metallic accents.
#
# Кластер 1:
#  - Документ 2: Streetwear is heavily influenced by skate and surf culture, often featuring distressed jeans, graphic T-shirts, caps, sneakers, and hoodies for a casual aesthetic.
#  - Документ 4: Urban style in modern times blends practicality with trends, emphasizing versatile items such as bomber jackets, joggers, and stylish backpacks.
#  - Документ 9: Haute couture represents the pinnacle of fashion design, emphasizing handcrafted, one-of-a-kind garments tailored to individual clients, often showcased on exclusive runways in Paris.
#  - Документ 12: Contemporary street style draws heavily from hip-hop culture, featuring oversized jackets, chunky sneakers, bold logos, and bright color palettes.
#  - Документ 13: Eco-friendly or sustainable fashion is a rising trend, emphasizing organic materials, ethical labor practices, and reducing environmental impact.
#
# Кластер 2:
#  - Документ 3: Art Nouveau fashion introduced flowing lines, floral motifs, and a focus on organic shapes, reflecting the broader artistic movement of the late 19th to early 20th century.
#  - Документ 5: Japanese avant-garde designers like Rei Kawakubo and Yohji Yamamoto redefined silhouettes with experimental cuts, challenging traditional definitions of shape and form.
#
# Кластер 3:
#  - Документ 1: Renaissance fashion in the 15th century was characterized by elaborate garments, rich fabrics, and detailed embroidery. Nobles wore doublets, hose, and ornate headdresses, symbolizing status and wealth.
#  - Документ 7: Victorian era fashion for women featured corsets, elaborate crinoline skirts, and high necklines, while men wore frock coats, waistcoats, and top hats.
#  - Документ 8: Minimalism in clothing embraces simple lines, neutral colors, and high-quality fabrics, reflecting a more functional and understated approach to fashion.
#  - Документ 11: Flapper dresses of the 1920s were straight and loose, featuring dropped waists, beaded fringes, and shorter hemlines, symbolizing women's changing social roles.
#  - Документ 14: In the 1960s, mod fashion popularized short hemlines, bold geometric patterns, and a youthful spirit. The shift dress became a symbol of modern femininity.
#  - Документ 15: The punk movement of the 1970s brought distressed clothing, leather jackets, safety pins, and a rebellious ethos, challenging mainstream fashion norms.

