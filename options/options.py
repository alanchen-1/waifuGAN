import argparse
import os

class TrainOptions():
    """
    Class for training options.
    Used for train.py.
    """
    def __init__(self):
        """
        Constructor for TrainOptions.
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.add_arguments(parser)
    
    def add_arguments(self, parser : argparse.ArgumentParser):
        """
        Adds arguments to the parser.
            Parameters:
                parser (argparse.ArgumentParser) : argument parser to initialize
            Returns:
                parser (argparse.ArgumentParser) : initialized argument parser
        """
        parser.add_argument('--dataroot', type=str, default='./data/cropped_images/', help='path to data root directory')
        parser.add_argument('--output_dir', type=str, default='./results/', help='results path')
        parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--image_size', type=int, default=64, help='image size')
        parser.add_argument('--no_soft_labels', action='store_true', help='do not use soft labels (use real labels)')
        parser.add_argument('--epochs', type=int, default=100, help="number of epochs to run for")
        parser.add_argument('--no_plots', action='store_true', help='do not plot anything')
        return parser

    def export_options(self, opt):
        """
        Exports the options to a file in <output_dir>.
            Parameters:
                opt : parsed options
        """
        string = 'using options: \n'
        for option, value in sorted(vars(opt).items()):
            string += f'{option} : {value}\n'
        string += '\n'
        print(string)
        
        # export options to file in output_dir
        output_dir = opt.output_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f'options.txt')
        with open(filename, 'wt') as open_file:
           open_file.write(string) 

    def parse(self):
        """
        Parses, exports, and returns the options.
        """
        # get options
        opt = self.parser.parse_args()
        self.opt = opt
        # export
        self.export_options(opt)
        # return parsed options
        return opt
