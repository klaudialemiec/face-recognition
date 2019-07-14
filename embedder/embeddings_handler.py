
def embeddings_for_faces(faces, model):
    faces = [normalize_faces(face) for face in faces]
    embeddings = model.create_images_embeddings(faces)
    return embeddings


def embeddings_for_central_faces(faces, model):
    faces = [normalize_faces(face) for face in faces]
    embeddings = model.create_images_embeddings(faces)
    return embeddings


def embedding_for_face(face, model):
    normalized = [normalize_faces(face)]
    return model.create_images_embeddings(normalized)[0]


def normalize_faces(faces):
    return faces / 255.
