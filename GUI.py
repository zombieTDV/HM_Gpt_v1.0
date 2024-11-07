from tkinter import *
from tkinter import filedialog
import pickle

from GPT import GPT_MODEL

class cache:
    def __init__(self) -> None:
        self.gpt = None

gpt_cache = cache()

class Parameter_select_window:
    def __init__(self, training_path) -> None:
        self.top_window = Toplevel()
        self.top_window.title('select parameter for the model')
        self.top_window.geometry("620x470")
        self.top_window.config(background = "white")
        
        self.training_path = training_path

        self.fields = 'save path', 'save name', 'batch sizes', 'max sequence len', 'emb dims', 'n heads', 'n layers', 'lr', 'optimization method', 'epochs'
        self.parameters = []
        
        self.loop()
        print(f'{training_path=}')
        
    def export_configs(self, config: list, path, name):
        with open(path+'/'+name+'.config', 'wb') as fp:
            pickle.dump(config, fp)
            
        print(f'export config success! -> path: {path}/{name}')
        
    def fetch(self, entries):
        self.parameters = []
        for entry in entries:
            field = entry[0]
            text  = entry[1].get()
            self.parameters.append(text)
        

    def makeform(self, root):
        entries = []
        for field in self.fields:
            row = Frame(root)
            lab = Label(row, width=15, text=field, anchor='w')
            ent = Entry(row)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            entries.append((field, ent))
        return entries
    
    def train_GPT_model(self):
        if self.parameters == []:#default
            print('GOING with default parameters!')
            path = "./pretrain data/"
            name = 'NNCT'
            batch_sizes=32
            max_sequence_len = 64
            emb_dims = 128
            n_heads = 16
            n_layers = 7
            lr=1e-2
            optimization_method='AdamW'
            epochs = 3000
            self.parameters = [path, name, batch_sizes, max_sequence_len, \
                emb_dims, n_heads, n_layers, lr, optimization_method, epochs, self.training_path]
            
            cache.gpt = GPT_MODEL(self.training_path, path+'/'+name, batch_sizes, max_sequence_len, emb_dims, n_heads, n_layers, lr, optimization_method)
            cache.gpt.forward_pass(epochs, datas_export=True,finetune=False, load_datas=False)
            
            self.export_configs(self.parameters, path, name)
        
        elif self.parameters:
            path, name, batch_sizes, max_sequence_len, \
                emb_dims, n_heads, n_layers, lr, optimization_method, epochs = self.parameters
                
            self.parameters.append(self.training_path)
            print(optimization_method)
            cache.gpt = GPT_MODEL(self.training_path , path+'/'+name, batch_sizes, max_sequence_len, emb_dims, n_heads, n_layers, lr, optimization_method)
            cache.gpt.forward_pass(epochs, datas_export=True,finetune=False, load_datas=False)
            
            self.export_configs(self.parameters, path, name)


    def loop(self):
        note = Label(self.top_window, 
                            text = "Manually set the parameters for the model or leave it empty to use the default parameters",wraplength=200, 
                            fg = "red")
        
        ents = self.makeform(self.top_window)
        self.top_window.bind('<Return>', (lambda event, e=ents: self.fetch(e)))   
        b1 = Button(self.top_window, text='Use new parameters',
                    command=(lambda e=ents: self.fetch(e)))
        b1.pack(side=LEFT, padx=5, pady=5)
        b2 = Button(self.top_window, text='Train', command=self.train_GPT_model)
        b2.pack(side=LEFT, padx=5, pady=5)
        
        note.pack(side=LEFT, padx=5, pady=5)
        
        
class GPT_GUI:
    def __init__(self) -> None:
        self.window = Tk()
        self.window.title('Simple GPT user interface')
        self.window.geometry("720x350")
        self.window.config(background = "white")
        
        self.training_path = ''
        self.config_path = ''
    
    
    def load_config(self, path): 
        with open (path, 'rb') as fp:
            config = pickle.load(fp)
            
        print(f'LOAD CONFIG SUCCESS! \n-> path: {path}')
        return config

    def import_pretrain(self, pretrain_path):
        path, name, batch_sizes, max_sequence_len, \
                emb_dims, n_heads, n_layers, lr, \
                    optimization_method, epochs, training_path = self.load_config(self.config_path)
        
        
        gpt_cache.gpt = GPT_MODEL(training_path, pretrain_path, batch_sizes, max_sequence_len, emb_dims, n_heads, n_layers, lr=0.001, optimization_method = 'GD')
        gpt_cache.gpt.forward_pass(epochs=150, datas_export=False,finetune=False, load_datas=True)
        
        print('Model import pretrain data completed!You can inferance now!\n')
        
    def model_inferance(self):
        try:
            gpt_cache.gpt.inferance()
        except Exception as e:
            print(e,'\n')
            print('You have to import pretrain data or train the new one in order to inferance the model!')
            
            
    def browse_config(self):
        filename = filedialog.askopenfilename(initialdir = "/",
                                            title = "Select config file",
                                            filetypes = (("Config file",
                                                            "*.*"),))
        self.config_path = filename
        
    def browse_pretrain_data(self):
        filename = filedialog.askopenfilename(initialdir = "/",
                                            title = "Select a File",
                                            filetypes = (("pretrain data file",
                                                            ".npz"),))
        pretrain_path = filename
        print(filename)
        self.import_pretrain(pretrain_path)
        
    def browse_file_for_training(self):
        filename = filedialog.askopenfilename(initialdir = "/",
                                            title = "Select a File",
                                            filetypes = (("Text files",
                                                            ".txt"),))
        self.training_path = filename
        self.second_win = Parameter_select_window(self.training_path)
	
    def __call__(self):
        label_file_explorer = Label(self.window, 
                                    text = "Choose whether you want to use a pretrain data(.npz) or train on your own with dataset(.txt)",
                                    width = 80, height = 4, 
                                    fg = "blue")
        
        button_browse_pretrain = Button(self.window, 
                                text = "Use pretrain data(.npz)",
                                command = self.browse_pretrain_data) 

        button_browse_file_for_training = Button(self.window, 
                                text = "Use data for training(.txt)",
                                command = self.browse_file_for_training) 
        
        button_browse_config = Button(self.window, 
                        text = "Browse config file(.config) correctly before use pretrain data",
                        command = self.browse_config, fg='red') 

        button_exit = Button(self.window, 
                            text = "Exit",
                            command = quit) 
        
        button_inferance = Button(self.window, 
                            text = "Inferance",
                            command = self.model_inferance) 

        
        label_file_explorer.grid(column = 1, row = 1, padx=5, pady=5)

        button_browse_pretrain.grid(column = 1, row = 2, padx=0, pady=0)
        button_browse_config.grid(column = 1, row = 3, padx=20, pady=20)
        button_browse_file_for_training.grid(column = 1, row = 4, padx=10, pady=10)
        

        button_inferance.grid(column=1, row=5, padx=20, pady=20)
        button_exit.grid(column = 1,row = 6, padx=5, pady=5)

        # Let the window wait for any events
        self.window.mainloop()
