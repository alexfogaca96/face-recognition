class SavedVariables():
    def __init__(self , i_h_weigths , h_o_weights , h_bias , o_bias ):
        self.i_h_weigths = i_h_weigths 
        self.h_o_weights = h_o_weights 
        self.h_bias = h_bias 
        self.o_bias = o_bias 
    
    def recover(self , mlp):
        mlp.i_h_weigths = self.i_h_weigths
        mlp.h_o_weights = self.h_o_weights 
        mlp.h_bias = self.h_bias
        mlp.o_bias = self.o_bias