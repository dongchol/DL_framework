from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model


class Model(nn.Module):
    def __init__(self, model_name, num_classes, feature_extract=True, use_pretrained=True):
        super(Model, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract      #if true we only update the reshaped layer params
        self.use_pretrained = use_pretrained        #if true we use pretrained model
        self.use_gpu = torch.cuda.is_available()    #for use cuda
        #self.criterion = criterion                  #loss
        #self.optimizer = optimizer                  #optimizer
        #self.scheduler = scheduler                  #scheduler, to control learning rate

    def initialize_model(self):
        model_ft = None
        input_size = 0

        if self.model_name == "resnet":
            model_ft = models.resnet152(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "SE_resnet":
            model_ft = ptcv_get_model("resnet50", pretrained=self.use_pretrained) #SE resnet18
            #self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.output.in_features
            #model_ft.children()
            model_ft.output = nn.Linear(num_ftrs, self.num_classes)
            #model_ft.output = nn.Linear(num_ftrs, 64)
            #model_ft.fc = nn.Linear(64, self.num_classes)
            #print(model_ft.dim())
            input_size = 224


        elif self.model_name == "alexnet":
            model_ft = models.alexnet(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,self.num_classes)
            input_size = 224

        elif self.model_name == "vgg":
            model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,self.num_classes)
            input_size = 224

        elif self.model_name == "squeezenet":
            model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = self.num_classes
            input_size = 224

        elif self.model_name == "densenet":
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "inception":
            model_ft = models.inception_v3(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        if self.use_gpu:
            model_ft = model_ft.cuda()

        return model_ft

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

