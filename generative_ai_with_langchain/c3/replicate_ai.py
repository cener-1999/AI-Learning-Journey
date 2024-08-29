from langchain_community.llms import Replicate

text2image = Replicate(
    model="adirik/flux-cinestill:216a43b9975de9768114644bbf8cd0cba54a923c6d0f65adceaccfc9383a938f",
    model_kwargs={"image_dimensions": "512x512"},)
image_url = text2image.invoke("a book cover for a book about creating generative ai applications in Python")
print(image_url)