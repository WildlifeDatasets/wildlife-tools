import os
import pickle


class Similarity:
    def __call__(self, query, database):
        raise NotImplementedError()

    def run_and_save(
        self, query, database, save_path, query_metadata=None, database_metadata=None
    ):
        similarities = self(query, database)

        os.makedirs(save_path, exist_ok=True)
        file_names = []
        for key, similarity in similarities.items():
            name = self.__class__.__name__ + "-" + str(key)
            data = {
                "name": key,
                "similarity": similarity,
                "metadata_query": query_metadata,
                "metadata_database": database_metadata,
            }

            file_name = os.path.join(save_path, name + ".pkl")
            with open(file_name, "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            file_names.append(file_name)

        return file_names
