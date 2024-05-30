import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizerFast
import pdb, tqdm
from torchsummary import summary
import transformers as tf
from torchvision import transforms
import kgbench as kg
from sklearn.decomposition import PCA
import sys
# from memory_profiler import profile
from torch.profiler import profile, record_function, ProfilerActivity



class SIMP_CNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 4)
        self.conv3 = nn.Conv2d(8, 16, 4)
        self.fc1 = nn.Linear(16 * 25 * 25, embed_size)

    def forward(self, x):
        # print("before first: ", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print("after first: ", x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print("after second: ", x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print("after third: ", x.size())
        x = torch.flatten(x, 1)
        # print("before fc: ", x.size())
        x = self.fc1(x)

        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        # padding = (kernel_size - 1) // 2
        padding = 0
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

def mobilenet_emb_e2e(embed_size, pretrained):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=pretrained)
    # model_fine = nn.Sequential(model.features, nn.Linear(in_features = 107520*7, out_features = embed_size, bias = True))
    model.classifier[1] = nn.Linear(model.last_channel, out_features=embed_size, bias=True)

    if torch.cuda.is_available():
        model.cuda()
    return model

def mobilenet_emb(pilimages, bs=512):

    # Create embeddings for image
    prep = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #-- Standard mobilenet preprocessing.

    image_embeddings = []

    if torch.cuda.is_available():
        cnnmodel.cuda()

    nimages = len(pilimages)
    imagegen = kg.to_tvbatches(pilimages, batch_size=bs, prep=prep, min_size=224, dtype=torch.float32)
    # pdb.set_trace()
    for batch in tqdm.tqdm(imagegen, total=nimages // bs):
        bn, c, h, w = batch.size()
        if torch.cuda.is_available():
            batch = batch.cuda()

        out = cnnmodel.features(batch)
        # pdb.set_trace()
        image_embeddings.append(out.view(bn, -1).to('cpu'))
        # print(image_embeddings[-1].size())

    return torch.cat(image_embeddings, dim=0)

def bert_emb(strings, bs_chars, mname='distilbert-base-cased'):
    # Sort by length and reverse the sort after computing embeddings
    # (this will speed up computation of the embeddings, by reducing the amount of padding required)

    indexed = list(enumerate(strings))
    indexed.sort(key=lambda p:len(p[1]))

    embeddings = bert_emb_([s for _, s in indexed], bs_chars)
    indices = torch.tensor([i for i, _ in indexed])
    _, iindices = indices.sort()

    return embeddings[iindices]

MNAME='distilbert-base-cased'
cnnmodel = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
bmodel = tf.DistilBertModel.from_pretrained(MNAME)
btok = tf.DistilBertTokenizerFast.from_pretrained(MNAME)

def bert_emb_(strings, bs_chars, ):

    pbar = tqdm.tqdm(total=len(strings))

    outs = []
    fr = 0
    while fr < len(strings):

        to = fr
        bs = 0
        while bs < bs_chars and to < len(strings):
            bs += len(strings[to])
            to += 1
            # -- add strings to the batch until it puts us over bs_chars

        # print('batch', fr, to, len(strings))
        strbatch = strings[fr:to]

        try:
            batch = btok(strbatch, padding=True, truncation=True, return_tensors="pt")
        except:
            print(strbatch)
            sys.exit()
        #-- tokenizer automatically prepends the CLS token
        inputs, mask = batch['input_ids'], batch['attention_mask']
        if torch.cuda.is_available():
            inputs, mask = inputs.cuda(), mask.cuda()

        if torch.cuda.is_available():
            bmodel.cuda()

        out = bmodel(inputs, mask)

        outs.append(out[0][:, 0, :].to('cpu')) # use only the CLS token

        pbar.update(len(strbatch))
        fr = to

    return torch.cat(outs, dim=0)


# @profile
def extract_embeddings(data, feat_size):
    with torch.no_grad():
        # pdb.set_trace()
        embeddings = []
        for datatype in data.datatypes():
            if datatype in ['iri', 'blank_node']:
                print(f'Initializing embedding for datatype {datatype}.')
                # create random embeddings
                # -- we will parametrize this part of the input later
                n = len(data.get_strings(dtype=datatype))
                nodes = torch.randn(n, feat_size)
                if torch.cuda.is_available():
                    nodes = nodes.cuda()

                embeddings.append(nodes)

            elif datatype == 'http://kgbench.info/dt#base64Image':
                print(f'Computing embeddings for images.')
                image_embeddings = mobilenet_emb(data.get_images(), bs=256)
                image_embeddings = pca(image_embeddings, target_dim=feat_size)
                embeddings.append(image_embeddings)

            else:
                # embed literal strings with DistilBERT
                print(f'Computing embeddings for datatype {datatype}.')
                string_embeddings = bert_emb(data.get_strings(dtype=datatype), bs_chars=50_000)
                string_embeddings = pca(string_embeddings, target_dim=feat_size)
                embeddings.append(string_embeddings)
        # pdb.set_trace()
        embeddings = torch.cat(embeddings, dim=0).to(torch.float)
        # -- note that we use the fact here that the data loader clusters the nodes by data type, in the
        #    order given by data._datasets


    # Split embeddings into trainable and non-trainable
    num_iri, num_bnode = len(data.datatype_l2g('iri')), len(data.datatype_l2g('blank_node'))
    numparms = num_iri + num_bnode
    trainable = embeddings[:numparms, :]
    constant = embeddings[numparms:, :]

    trainable = nn.Parameter(trainable)

    if torch.cuda.is_available():
        print('Using cuda.')
        trainable = trainable.cuda()
        constant = constant.cuda()
    embed_X = torch.cat([trainable, constant], dim=0)
    pdb.set_trace()
    global bmodel
    global cnnmodel
    del bmodel
    del cnnmodel
    torch.cuda.empty_cache()
    # print("hiiiiiiiii",  sys.getsizeof(bmodel), " hiiiiiii")
    # pdb.set_trace()
    return embed_X

def pca(tensor, target_dim):
    """
    Applies PCA to a torch matrix to reduce it to the target dimension
    """

    n, f = tensor.size()
    if n < 25:  # no point in PCA, just clip
        res = tensor[:, :target_dim]
    else:
        if tensor.is_cuda:
            tensor = tensor.to('cpu')
        model = PCA(n_components=target_dim, whiten=True)

        res = model.fit_transform(tensor)
        res = torch.from_numpy(res)

    if torch.cuda.is_available():
        res = res.cuda()

    return res

