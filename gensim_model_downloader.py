import gensim.downloader as api

def list_available_gensim_models():

    # retrieves and prints the list of available models in Gensim model repository
    available_models = api.info()['models'].keys()
    print(list(available_models))

def download_and_save_model(model_name, file_name):
    # downloads a specified Gensim model and saves it locally with the given file name
    model = api.load(model_name)
    model.save(file_name)
    print(f"Model '{model_name}' saved as '{file_name}'.")
