

import torch
import numpy as np

from .params import featureMatchingOutputs, featureMatchingParameters
import psutil

import tqdm

class featureMatching:

    def __init__(self, params: featureMatchingParameters):

        self._params = params


    def compute_delta(self, dissimilarities):
        # dissimilarities: (N1, N2, Ncells)
        # Compute the moments
        mu_tot   = dissimilarities.mean(dim=1).sum(dim=1)
        var_tot  = dissimilarities.var(dim=1).sum(dim=1)

        # hypothesis: sum(dissimilarities, dim=-1) ~ N(mu_tot, var_tot)

        if self._params.distribution == 'normal':
            from torch.special import log_ndtr # log(phi(x))
            standardized = (dissimilarities.sum(-1) - mu_tot[:, None]) / (var_tot[:, None]**.5 + 1e-15)
            nlfa = - log_ndtr(standardized)

        return nlfa


    def match(self, query_histograms, key_histograms, display_progress=False):
        device = query_histograms.device
        N1, _, _ = query_histograms.shape
        N2, Nh, Nbins = key_histograms.shape

        log_quantile = np.log(self._params.epsilon) - np.log(N1) - np.log(N2)
        nlfa_threshold = -log_quantile

        if not self._params.partial_output:
            total_dissimilarities = torch.zeros((N1, N2), device='cpu')
            nlfa = torch.zeros_like(total_dissimilarities)

        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0) if device == 'cuda' else psutil.virtual_memory().available
        available_memory -= 100 * 2**20 # remove 100 MB to be sure everything fits in memory
        element_size = query_histograms.element_size()

        slice_memory = N2 * Nh * Nbins * element_size * 20

        # Compute the batch size as the maximum amount of (N2, Nh, Nbins) slices we can fit in memory
        batch_size = max(1, available_memory // slice_memory)
        # batch_size = 1
        print('Batch size : ', batch_size)

        matches_list = []

        for idx_start in tqdm.tqdm(range(0, N1, batch_size)):
            queries = query_histograms[idx_start:idx_start+batch_size]

            # Compute the N_cells dissimilarities for the batch of queries
            dissimilarities = self.compute_dissimilarities(queries, key_histograms) # N1, N2, N_cells

            # Compute total dissimilarities D(a^i, b^j) for all pairs
            D = dissimilarities.sum(dim=-1)  # (N1, N2)

            nlfa_batch = self.compute_delta(dissimilarities)

            if not self._params.partial_output:
                nlfa[idx_start:idx_start+batch_size] = nlfa_batch
                total_dissimilarities[idx_start:idx_start+batch_size] = D

            # Find meaningfull matches 
            matches = nlfa_batch >= nlfa_threshold

            # Get match indices (query_idx, candidate_idx)
            match_indices = torch.nonzero(matches, as_tuple=False)  # (num_matches, 2)
            match_indices[:, 0] += idx_start
            matches_list.append(match_indices.cpu())

        matches = torch.cat(matches_list, dim=0)

        if self._params.partial_output:
            return matches
        
        return matches, nlfa, total_dissimilarities
    

    def nlfa_theshold(self, query_histograms, key_histograms):
        N1, _, _ = query_histograms.shape
        N2, Nh, Nbins = key_histograms.shape
        log_quantile = np.log(self._params.epsilon) - np.log(N1) - np.log(N2)
        return -log_quantile


    def __call__(self, query_histograms, key_histograms):
        nlfa_threshold = self.nlfa_theshold(query_histograms, key_histograms)
        matches1, nlfa1, dissim = self.match(query_histograms, key_histograms)

        if not self._params.reciprocal_only:
            return featureMatchingOutputs(
                match_indices=matches1,
                nlfa=nlfa1,
                dissimilarities=dissim,
                nlfa_threshold=nlfa_threshold
            )
        
        _, nlfa2, _ = self.match(key_histograms, query_histograms)
    
        matches = torch.logical_and(nlfa1 >= nlfa_threshold,  nlfa2.T >= nlfa_threshold)
        match_indices = torch.nonzero(matches, as_tuple=False)

        return featureMatchingOutputs(
            match_indices=match_indices,
            nlfa=nlfa1,
            dissimilarities=dissim,
            nlfa_threshold=nlfa_threshold,
            nlfa2=nlfa2.T
        )


    def compute_dissimilarities(self, queries, keys):
        """
        Compute dissimilarities between queries and keys.
        
        Args:
            queries: (n_q, Nh, Nbins) or (n_q, n_k, Nh, Nbins) for broadcasting
            keys: (n_k, Nh, Nbins) or (n_q, n_k, Nh, Nbins) for element-wise
        
        Returns:
            dissimilarities: (n_q, n_k, Nh) or (n_q, Nh) depending on input shapes
        """
        if self._params.metric == "L2":
            if queries.dim() == 3 and keys.dim() == 3:
                # All-pairs: (n_q, Nh, Nbins) vs (n_k, Nh, Nbins)
                return torch.pow(queries[:, None] - keys[None, :], 2).sum(-1)
            else:
                # Element-wise: (n, Nh, Nbins) vs (n, Nh, Nbins)
                return torch.pow(queries - keys, 2).sum(-1)
                
        elif self._params.metric == 'CEMD':
            if queries.dim() == 3 and keys.dim() == 3:
                # All-pairs computation
                X = queries.cumsum(-1)[:, None] - keys.cumsum(-1)[None, :]  # (n_q, n_k, Nh, Nbins)
            else:
                # Element-wise computation
                X = queries.cumsum(-1) - keys.cumsum(-1)  # (n, Nh, Nbins) or (n_q, n_k, Nh, Nbins)
            
            # Compute ||X_k||_1 for each k
            X_padded = torch.cat([torch.zeros_like(X[..., :1]), X], dim=-1)  # ..., Nbins+1
            
            l1_norms = []
            for starting_bin in range(X.shape[-1] - 1):  # -1 because X_padded has Nbins+1
                X_k = X - X_padded[..., starting_bin:starting_bin+1]  # Subtract X[k-1] from all positions
                l1_norm = X_k.abs().sum(dim=-1) / X.shape[-1]
                l1_norms.append(l1_norm)    
            
            cemd = torch.stack(l1_norms, dim=-1).min(dim=-1).values
            return cemd


    def match_subset(self, query_histograms, key_histograms, match_indices, deltas):
        """
        Verify matches for a specific subset of query-key pairs using batching.
        
        Args:
            query_histograms: (N, Nh, Nbins) query histograms
            key_histograms:   (N, Nh, Nbins) keys
            match_indices:    (N_matches, 2) pairs to test (query_idx, key_idx)
            deltas:           (N,) threshold deltas for each query
        
        Returns:
            valid_matches: (N_valid, 2) indices of matches that satisfy the delta criterion
        """
        if len(match_indices) == 0:
            return match_indices
        
        device = query_histograms.device
        _, Nh, Nbins = query_histograms.shape
        
        # Calculate available memory and batch size
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0) if device.type == 'cuda' else psutil.virtual_memory().available
        available_memory -= 100 * 2**20  # Remove 100 MB safety margin
        element_size = query_histograms.element_size()
        
        # Memory needed per match: 2 histograms (query + key) + dissimilarities + intermediate computations
        memory_per_match = 2 * Nh * Nbins * element_size + Nh * element_size
        if self._params.metric == 'CEMD':
            memory_per_match *= 10  # CEMD needs more memory for intermediate tensors
        else:
            memory_per_match *= 3   # L2 needs less overhead
        
        batch_size = max(1, available_memory // memory_per_match)
        
        print(f'match_subset batch size: {batch_size:,} (processing {len(match_indices):,} total matches)')
        
        valid_matches_list = []
        n_matches = len(match_indices)
        
        for idx_start in tqdm.tqdm(range(0, n_matches, batch_size), desc="Verifying matches"):
            idx_end = min(idx_start + batch_size, n_matches)
            batch_indices = match_indices[idx_start:idx_end]
            
            # Extract the specific pairs we need to check
            query_indices = batch_indices[:, 0]  # (batch_size,)
            key_indices = batch_indices[:, 1]    # (batch_size,)
            
            # Get the corresponding histograms
            queries = query_histograms[query_indices]  # (batch_size, Nh, Nbins)
            keys = key_histograms[key_indices]         # (batch_size, Nh, Nbins)
            
            # Compute dissimilarities for each pair (element-wise)
            dissimilarities = self.compute_dissimilarities(queries, keys)  # (batch_size, Nh)
            
            # Compute total dissimilarity for each pair
            D = dissimilarities.sum(dim=-1)  # (batch_size,)
            
            # Get the corresponding deltas for each query
            delta_values = deltas[query_indices]  # (batch_size,)
            
            # Keep only matches where D <= delta
            valid_mask = D <= delta_values  # (batch_size,)
            valid_batch_matches = batch_indices[valid_mask]  # (N_valid_batch, 2)
            
            valid_matches_list.append(valid_batch_matches)
            
            # Clean up to free memory
            del queries, keys, dissimilarities, D, delta_values, valid_mask
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Concatenate all valid matches
        valid_matches = torch.cat(valid_matches_list, dim=0) if valid_matches_list else match_indices[:0]
        
        return valid_matches




