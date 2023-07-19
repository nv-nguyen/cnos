from torch import nn
import torch
from src.poses.utils import load_rotation_transform, convert_openCV_to_openGL_torch
import torch.nn.functional as F
from src.model.utils import BatchedData


class Similarity(nn.Module):
    def __init__(self, metric="cosine", chunk_size=64):
        super(Similarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query, reference):
        query = F.normalize(query, dim=-1)
        reference = F.normalize(reference, dim=-1)
        similarity = F.cosine_similarity(query, reference, dim=-1)
        return similarity.clamp(min=0.0, max=1.0)


class PairwiseSimilarity(nn.Module):
    def __init__(self, metric="cosine", chunk_size=64):
        super(PairwiseSimilarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query, reference):
        N_query = query.shape[0]
        N_objects, N_templates = reference.shape[0], reference.shape[1]
        references = reference.clone().unsqueeze(0).repeat(N_query, 1, 1, 1)
        queries = query.clone().unsqueeze(1).repeat(1, N_templates, 1)
        queries = F.normalize(queries, dim=-1)
        references = F.normalize(references, dim=-1)

        similarity = BatchedData(batch_size=None)
        for idx_obj in range(N_objects):
            sim = F.cosine_similarity(
                queries, references[:, idx_obj], dim=-1
            )  # N_query x N_templates
            similarity.append(sim)
        similarity.stack()
        similarity = similarity.data
        similarity = similarity.permute(1, 0, 2)  # N_query x N_objects x N_templates
        return similarity.clamp(min=0.0, max=1.0)
