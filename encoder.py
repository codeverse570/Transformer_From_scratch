class Encoder:
    def __init__(self,d_model,h_count,d_ff):
      self.positional_encoding=[]
      self.encodings=[]
      self.d_model=d_model
      self.h_count=h_count
      self.d_ff=d_ff
      self.d_k= d_model/h_count

    def create_embeddings(self,words):
        