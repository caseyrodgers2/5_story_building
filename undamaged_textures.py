# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:12:04 2022

@author: Casey Rodgers

References for creating the RGB map: 
    -A. Efros and W. T. Freeman, "Image quilting for texture synthesis and transfer", Proc. ACM Conf. Computer Graphics (SIGGRAPH), pp. 341-346, 2001-Aug.  
    -Wang, Yuxiong. “Programming Project #3: Gradient-Domain Fusion CS445: Computational Photography.” Computational Photography (CS 445) - Fall 2021, https://yxw.cs.illinois.edu/course/CS445/F21/projects/gradient/ComputationalPhotography_ProjectGradient.html. 
    -P. Pérez, "Poisson Image Editing", Proc. ACM Siggraph, 2003. 
    
Reference for normal map: 
    -“Whats the Logic behind Creating a Normal Map from a Texture?” Stack Overflow, 1 July 1960, https://stackoverflow.com/questions/10652797/whats-the-logic-behind-creating-a-normal-map-from-a-texture. 
    -“Calculate Normals from Heightmap.” Stack Overflow, 1 May 1966, https://stackoverflow.com/questions/49640250/calculate-normals-from-heightmap. 
    -201, Lethal Raptor Games, et al. “Calculate Normal's from Heightmap.” Ultra Engine Community, 13 May 2017, https://www.ultraengine.com/community/topic/16244-calculate-normals-from-heightmap/. 
    -Shaffer, Eric. “MP2: Terrain Modeling.” CS 418 Interactive Computer Graphics, https://illinois-cs418.github.io/assignments/mp2.html. 
    -“How to Normalize a NumPy Array to a Unit Vector?” Stack Overflow, 1 Feb. 1962, https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector. 

    
"""
import cv2
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import random
import scipy
import scipy.sparse.linalg
from PIL import Image


class UndamagedTex():

    ## Create random concrete patch from sample texture
    def choose_random(self, sample, patch_size_w, patch_size_h):
        """
        Randomly chooses a patch from the sample of size patch_size x patch_size.
    
        Inputs:
        sample: numpy.ndarray Sample image
        patch_size_w: int Width of patch
        patch_size_h: int Height of patch
    
        Returns random patch
        """
    
        # Choose a random patch
        h, w, d = sample.shape # Sample shape
        rand_coords = np.random.rand(2, 1) # Random numbers btwn 0 and 1 for coords
        rand_x = int(rand_coords[0] * (h - patch_size_h)) # random x coord
        rand_y = int(rand_coords[1] * (w - patch_size_w)) # random y coord
    
        patch_x = rand_x # Patch x index
        patch_y = rand_y # Patch y index
    
        patch = sample[patch_x:patch_x+patch_size_h, patch_y:patch_y+patch_size_w, :] # Extract the patch
    
        return patch
    
    
    
    ## Perform template matching
    def ssd_patch(self, template, sample):
        """
        Performs template matching with the overlapping region, computing the cost of sampling each patch, based on the
        sum of squared differences (SSD) of the overlapping regions of the existing and sampled patch.
    
        Inputs:
        template: numpy.ndarray Unfinished patch with overlap from output image
        sample: numpy.ndarray Sample image
    
        Returns SSD Cost as an numpy.ndarray of image
        """
    
        template2 = template.copy() # Template
        sample2 = sample.copy() # Sample Image
    
        ssd_cost = np.float64(np.zeros(sample.shape)) # Initialize ssd_cost
    
        ## Create a mask from the template
        mask2 = template.copy() # Initialize mask
        #if (len(sample.shape) == 3):
        mask2[mask2 > 0] = 255 # Set non-zero values of array to 1
        #plt.figure()
        #plt.imshow(mask2)
    
        # RGB Image
        if (len(sample.shape) == 3):
            for i in range(3):
                M = np.float64(mask2[:, :, i] / 255.0) # Mask
                T = np.float64(template2[:, :, i] / 255.0) # Template
                I = np.float64(sample2[:, :, i] / 255.0) # Sample image
                
                part1 = ((M*T)**2).sum() - 2 * cv2.filter2D(I, ddepth=-1, kernel=M*T)
                part2 = cv2.filter2D(I**2, ddepth=-1, kernel=M)
                ssd_cost[:, :, i] = part1 + part2
                
            ssd_cost_tot = np.sum(ssd_cost, axis=2)
    
        # Black and White Image
        else:
            M = np.float64(mask2[:, :] / 255.0) # Mask
            T = np.float64(template2[:, :] / 255.0) # Template
            I = np.float64(sample2[:, :] / 255.0) # Sample image
            
            part1 = ((M*T)**2).sum() - 2 * cv2.filter2D(I, ddepth=-1, kernel=M*T)
            part2 = cv2.filter2D(I**2, ddepth=-1, kernel=M)
            ssd_cost[:, :] = part1 + part2
    
            ssd_cost_tot = ssd_cost
    
        #print(ssd_cost_tot.shape)
    
        return ssd_cost_tot
    
    
    
    ## Choose a random sample from a group of samples with the lowest costs
    def choose_sample(self, ssd_cost, sample, patch_size_w, patch_size_h, tol):
        """
        Select a randomly sampled patch from pool of size tol of the lowest costs
    
        Inputs:
        ssd_cost: numpy.ndarray SSD cost 2D array for the template
        sample: numpy.ndarray Sample image
        patch_size_w: int Patch size
        patch_size_h: int Patch size
        tol: float Tolerance. Top tol patches create the pool that is randomly pulled from for the patch
    
        Returns random patch
        """
    
        ## Prepare SSD
        h, w, d = sample.shape # Sample shape
        ssd_cost_copy = ssd_cost.copy() # Create copy of ssd_cost
    
        start1 = int(patch_size_h/2) # Find start of non-zero sample
        start2 = int(patch_size_w/2) # Find start of non-zero sample
        end1 = h - int(patch_size_h/2) - 1 # Find end of non-zero sample.
        end2 = w - int(patch_size_w/2) - 1 # Find end of non-zero sample.
    
        ssd_cost_copy = ssd_cost_copy[start1:end1, start2:end2] # SSD cost non-zero values
        #print(ssd_cost_copy)
        #print(ssd_cost_copy.shape)
        #print(sample.shape)
    
    
        ## Sort ssd_cost
        ssd_cost_vec = ssd_cost_copy.flatten() # Flatten from 2D array to 1D array
        sorted_ind = np.argsort(ssd_cost_vec, None) # Sort 1D ssd_cost vector
        sort_r, sort_c = np.unravel_index(sorted_ind, ssd_cost_copy.shape) # Get 2D indices from 1D index
    
    
        ## Randomly choose from top tol lowest cost
        r = int(random.random() * tol) # Get random integer from 0 to tol - 1
        select_r = sort_r[r] # Selected row index of chosen patch
        select_c = sort_c[r] # Selected col index of chosen patch
    
    
        ## Extract patch
        start_x = select_r # Starting x ind
        start_y = select_c # Starting y ind
    
        #print(start_x, start_y)
    
        patch = sample[start_x:start_x+patch_size_h, start_y:start_y+patch_size_w, :]
    
        return patch
    
    
    
    def cut(self, err_patch):
        """
        Compute the minimum path frm the left to right side of the patch
        
        :param err_patch: numpy.ndarray    cost of cutting through each pixel
        :return: numpy.ndarray             a 0-1 mask that indicates which pixels should be on either side of the cut
        
        Function from: 
        Wang, Yuxiong. “Programming Project #3: Gradient-Domain Fusion CS445: Computational Photography.” Computational Photography (CS 445) - Fall 2021, https://yxw.cs.illinois.edu/course/CS445/F21/projects/gradient/ComputationalPhotography_ProjectGradient.html. 
        
        """
        # create padding on top and bottom with very large cost
        padding = np.expand_dims(np.ones(err_patch.shape[1]).T*1e10,0)
        err_patch = np.concatenate((padding, err_patch, padding), axis=0)
        h, w = err_patch.shape
        path = np.zeros([h,w], dtype="int")
        cost = np.zeros([h,w])
        cost[:,0] = err_patch[:, 0]
        cost[0,:] = err_patch[0, :]
        cost[cost.shape[0]-1,:] = err_patch[err_patch.shape[0]-1, :]
        
        # for each column, compute the cheapest connected path to the left
        # cost of path for each row from left upper/same/lower pixel
        for x in range(1,w):
            # cost of path for each row from left upper/same/lower pixel
            tmp = np.vstack((cost[0:h-2,x-1], cost[1:h-1, x-1], cost[2:h, x-1]))
            mi = tmp.argmin(axis=0)
            path[1:h-1, x] = np.arange(1, h-1, 1).T + mi # save the next step of the path
            cost[1:h-1, x] = cost[path[1:h-1, x] - 1, x-1] + err_patch[1:h-1, x]

        path = path[1:path.shape[0]-1, :] - 1
        cost = cost[1:cost.shape[0]-1, :]
        
        # create the mask based on the best path
        mask = np.zeros(path.shape, dtype="int")
        best_path = np.zeros(path.shape[1], dtype="int")
        best_path[len(best_path)-1] = np.argmin(cost[:, cost.shape[1]-1]) + 1
        mask[0:best_path[best_path.shape[0]-1], mask.shape[1]-1] = 1
        for x in range(best_path.size-1, 0, -1):
            best_path[x-1] = path[best_path[x]-1, x]
            mask[:best_path[x-1], x-1] = 1
        mask ^= 1
        return mask
    
    
    
    ## Cut an error patch along the min cost path
    def customized_cut(self, e1, e2, vertical):
        """
        Cut error patch along min cost path
    
        Inputs: 
        e1: np.ndarray First block overlap
        e2: np.ndarray Second block overlap
        vertical: int Is the cut vertical? 0 = no. 1 = yes
    
        Returns 3d mask array of min cut
        """
    
        #print(e1.shape, e2.shape)
    
        err_patch_tot = np.float64(np.zeros(e1.shape)) # Initialize the error patch
    
        # Find error patch for each channel
        for i in range(3):
            part1 = np.float64(e1[:, :, i] / 255.0) # Part 1
            part2 = np.float64(e2[:, :, i] / 255.0) # Part 2
    
            err_patch_tot[:, :, i] = (part1 - part2)**2 # Square of difference 
    
        err_patch = np.sum(err_patch_tot, axis=2) # Sum error patch values
        #print(err_patch.shape)
    
        #plt.figure()
        #plt.imshow(err_patch)
    
        mask = np.zeros(err_patch.shape) # Initialize mask
    
        if vertical == 0:
            mask = self.cut(err_patch) # Horizontal cut
        else:
            mask = self.cut(np.transpose(err_patch)) # Vertical cut
            mask = np.transpose(mask)
    
        # Turn mask to 3d mask for element-wise multiplication later
        mask_3d = np.zeros(e1.shape) # Create a 3d mask array for element multiplication
        
        for i in range(3):
            mask_3d[:, :, i] = mask
    
        return mask_3d
    
    
    ## Sample and quilt the patches together
    def quilt_cut(self, sample, out_size_w, out_size_h, patch_size_w, patch_size_h, overlap_w, overlap_h, tol):
        """
        Samples square patches of size patchsize from sample using seam finding in order to create an output image 
        of size output
        Feel free to add function parameters
        :param sample: numpy.ndarray
        :param out_size_w: int
        :param out_size_h: int
        :param patch_size_w: int
        :param patch_size_h: int
        :param overlap_w: int
        :param overlap_h: int
        :param tol: float
        :return: numpy.ndarray
        """
    
        ## Create a blank black image
        output = np.uint8(np.zeros((out_size_h, out_size_w, 3)))
    
        ## Do we need a border?
        remainder_x = (out_size_h - patch_size_h) % (patch_size_h - overlap_h)
        remainder_y = (out_size_w - patch_size_w) % (patch_size_w - overlap_w)
        start_x0 = int(remainder_x / 2) # Add black border at top of size remainder/2
        start_y0 = int(remainder_y / 2) # Add black border at left of size remainder/2
    
        start_x = start_x0 # # X position where to start the patch
        start_y = start_y0 # Y position where to start the patch
    
        ## Keep adding random patches until the out image is full
        while (start_x + patch_size_h <= out_size_h):
    
            while (start_y + patch_size_w <= out_size_w):
    
                ## Find the patch
                # If it's the top left patch, then choose a random patch
                if (start_x == start_x0 and start_y == start_y0):
                    patch = self.choose_random(sample, patch_size_w, patch_size_h)
                    #patch = sample[0:patch_size_w, 0:patch_size_h, :] # Sanity check
    
                # Otherwise, we need to perform ssd cost analysis
                else:
                    # Extract curr output patch
                    template = output[start_x:start_x+patch_size_h, start_y:start_y+patch_size_w, :] 
                    #plt.figure()
                    #plt.imshow(template)
                    
                    ssd_cost = self.ssd_patch(template, sample) # Find the ssd cost
    
                    patch_0 = self.choose_sample(ssd_cost, sample, patch_size_w, patch_size_h, tol) # Get patch
                    #plt.figure()
                    #plt.imshow(patch_0)
    
                    # Find overlapped region
                    template2 = template.copy()
                    patch = patch_0.copy()
    
    
                    # Top row (only vertical cut)
                    if (start_x == start_x0):
    
                        # Find mask
                        e1 = template2[:, 0:overlap_w, :] # Template
                        #plt.figure()
                        #plt.imshow(e1)
                        
                        e2 = patch_0[:, 0:overlap_w, :] # Patch
                        #plt.figure()
                        #plt.imshow(e2)
                        
                        mask = self.customized_cut(e1, e2, 1) # Mask
                        #plt.figure()
                        #plt.imshow(mask)
                        #print(mask)
                        
                        # Combine to find patch
                        part_patch = np.multiply(e2, mask) # Part from patch
                        #plt.figure()
                        #plt.imshow(part_patch)
                        
                        part_template = np.multiply(e1, np.logical_not(mask)) # Part from template
                        #plt.figure()
                        #plt.imshow(part_template)
                        
                        patch[:, 0:overlap_w, :] = part_patch + part_template
    
                    # First column (only horizontal cut)
                    elif (start_y == start_y0):
                        e1 = template2[0:overlap_h, :, :] # Template
                        e2 = patch_0[0:overlap_h, :, :] # Patch
                        mask = self.customized_cut(e1, e2, 0) # Mask
    
                        # Combine to find patch
                        part_patch = np.multiply(e2, mask) # Part from patch
                        part_template = np.multiply(e1, np.logical_not(mask)) # Part from template
    
                        patch[0:overlap_h, :, :] = part_patch + part_template
    
    
                    # Everything else (both a vertical and horizontal cut)
                    else:
    
                        #plt.figure()
                        #plt.imshow(template2)
                        #plt.figure()
                        #plt.imshow(patch_0)
                        # Vertical cut
                        
                        e1 = template2[:, 0:overlap_w, :] # Template
                        e2 = patch_0[:, 0:overlap_w, :] # Patch
                        mask1 = self.customized_cut(e1, e2, 1) # Mask
    
                        # Horizontal cut
                        e3 = template2[0:overlap_h, :, :] # Template
                        e4 = patch_0[0:overlap_h, :, :] # Patch
                        mask2 = self.customized_cut(e3, e4, 0) # Mask
    
                        # Combine masks
                        patch_mask1 = np.zeros(patch_0.shape) # Initialize patch mask 1
                        patch_mask1[:, 0:overlap_w, :] = np.logical_not(mask1) # Add mask 1
                        #plt.figure()
                        #plt.imshow(patch_mask1)
                        
                        patch_mask2 = np.zeros(patch_0.shape) # Initialize patch mask 2
                        patch_mask2[0:overlap_h, :, :] = np.logical_not(mask2) # Add mask 2
                        #plt.figure()
                        #plt.imshow(patch_mask2)
                        
                        patch_mask = np.logical_or(patch_mask1, patch_mask2) # Patch mask total
    
                        patch_mask2 = np.multiply(np.ones(patch_mask.shape), patch_mask) # For the figure
                        #plt.figure()
                        #plt.imshow(patch_mask2)
                        
                        # Combine to find patch
                        part_patch = np.multiply(template.copy(), patch_mask) # Template part
                        #plt.figure()
                        #plt.imshow(part_patch)
                        
                        part_template = np.multiply(patch_0.copy(), np.logical_not(patch_mask)) # Patch part
                        #plt.figure()
                        #plt.imshow(part_template)
                        
                        patch = part_patch + part_template # Add patches together
    
    
                # Add patch to output image
                output[start_x:start_x+patch_size_h, start_y:start_y+patch_size_w, :] = patch # Add patch to output imag
                #plt.figure()
                #plt.imshow(output)
    
                # Update start variables
                start_y = start_y + patch_size_w - overlap_w
                #print(start_y)
    
            start_x = start_x + patch_size_h - overlap_h
            start_y = start_y0
            #print(start_x)
    
        ## Return image
        return output 
    
    
    
    ## Create a texture of a specified size with a concrete sample image
    def createTexture(self, conc_sample_file_name, w, h, actW, conc_act_w):
        """ 
        Create an image of repeated texture that's some size
        conc_sample = concrete sample image file name
        w = image width
        h = image height
        actW = actual width of structure (m)
        conc_act_w = concrete actual width (m) 
        Returns texture image as np.ndarray
        """
        
        # Figure out new dimensions
        conc_tex1 = Image.open(conc_sample_file_name)
        conc_w, conc_h = conc_tex1.size             # texture size
        new_width = int(w / actW * conc_act_w)      # new tex width
        new_height = int(conc_h / conc_w * new_width)   # new tex height 
        conc_tex1 = conc_tex1.resize((new_width, new_height))
        #print(new_width)
        #print(new_height)
            
        # Paste enough concrete textures to cover cropped image
        concrete = np.zeros((h, w),np.uint8)
        concrete = Image.fromarray(concrete)
        concrete = concrete.convert("RGB")
        pasteX = 0
        pasteY = 0
        while (pasteX < w):
            #print(pasteX)
            while (pasteY < h):
                concrete.paste(conc_tex1, (pasteX, pasteY))
                #concrete.show()
                #print(pasteY)
                pasteY = pasteY + new_height
            pasteX = pasteX + new_width
            pasteY = 0
            
        # Converts image np.ndarray
        concrete = np.array(concrete)
        concrete = concrete[:, :, ::-1].copy()  # Convert to BGR
        
        # Close open image and return concrete
        conc_tex1.close()
        return concrete
    
    
    
    ## Make the texture seamless
    def seamless(self, im):
     
        """
        Converts non-seamless texture to be seamless. This works for only ONE channel.
        A for-loop calling this function for each channel must be done for an RGB image.
    
        :param im: numpy.ndarray
        """
    
        # Image Shape and variable coords set up
        #print(im)
        im_h, im_w = im.shape # Image shape
        im2var = np.arange(im_h * im_w).reshape(im_h, im_w) # Variable array
        #print(im2var.shape)
    
        # Stuff for Equations
        e = 0 # Equation counter
        neq = im_h * (im_w-1) + (im_h-1) * im_w  # Number of equations for derivatives
        neq = neq + 2 * im_w + 2 * im_h          # Add number of equations for border
        #neq = neq + 1                            # Add number of equations for color in middle of image
    
        # Intialize matrices for equation
        A = scipy.sparse.lil_matrix((neq, im_h * im_w), dtype='double') # Initialize lil matrix A
        b = np.zeros((neq, 1), dtype='double') # Initialize known matrix b
    
        #print(A.get_shape())
    
        # Actually fill matrices by going through each pixel
        for y in range(im_h):
            for x in range(im_w):
                #print(x)
                #print(y)
                #print(im2var[y][x+1])
    
                # Objective 1 (derivative for across)
                if not(x == im_w - 1):
                    A[e, im2var[y][x+1]] = 1
                    A[e, im2var[y][x]] = -1
                    b[e] = im[y][x+1] - im[y][x] # Gradient
                    e = e + 1
                    
                # For last column, we want the east border to match the west border
                else:
                    avg_pix = 0.5 * (im[y][x] + im[y][0]) # Find average of west and east border
                    
                    # Set west border
                    A[e, im2var[y][x]] = 1
                    b[e] = avg_pix
                    e = e + 1
                    
                    # Set east border
                    A[e, im2var[y][0]] = 1
                    b[e] = avg_pix
                    e = e + 1
                    
                
                # Objective 2 (derivative for down)
                if not(y == im_h - 1):
                    A[e, im2var[y+1][x]] = 1
                    A[e, im2var[y][x]] = -1
                    b[e] = im[y+1][x] - im[y][x] # Gradient
                    e = e + 1
                    
                # For last row, we want the north border to match the south border
                else:
                    avg_pix = 0.5 * (im[y][x] + im[0][x]) # Find average of west and east border
                    
                    # Set south border
                    A[e, im2var[y][x]] = 1
                    b[e] = avg_pix
                    e = e + 1
                    
                    # Set north border
                    A[e, im2var[0][x]] = 1
                    b[e] = avg_pix
                    e = e + 1
                    
                    
        # Objective 3 (Color in middle of image stays the same)
        #ind1 = np.int64(im_h/2) # y index of a pixel in the middle of the image
        #ind2 = np.int64(im_w/2) # x index of a pixel in the middle of the image
        #A[e, im2var[ind1][ind2]] = 1
        #b[e] = im[ind1][ind2] # Source image pixel
    
        # Solve system of linear equations
        v = scipy.sparse.linalg.lsqr(A.tocsr(), b); # Solve w/ csr
        v_arr = v[0] # Extract v array
    
        # Turn variable vector into image and return
        im_out = v_arr.reshape(im_h, im_w)
        #print(im_out.shape)
    
        return im_out
    
    
    ## Normalize a vector (helper function for the normal map)
    def normalize(self, vector):
        """
        Normalize a vector
        vector = vector to normalize as np.array
        Returns normalized vector
        """
        
        norm = la.norm(vector)  # Find length of vector
        if norm == 0:           # If length of vector = 0, then return vector
            return vector
        return vector / norm    # Else, return normalized vector
    
    
    
    ## Create a normal map
    def createNormalMap(self, displ_img):
        """
        Create a normal map from a displacement map or grayscale image
        displ_img = GRAYscale image as image (assume this is a seamless image)
        Returns normal map as cv array
        """
        
        displ_arr = displ_img.astype(np.float64) / 255  # Convert to float
        #print(displ_arr)
        (height, width) = displ_arr.shape         # Shape
        #print("width={}, height={}".format(width, height))
        
        normal = np.zeros((height, width, 3))        # Empty normal map
        faceNorms = np.ones((height, 2*width, 3))   # 2 faces per square
        faceNorms = -1 * faceNorms                  # -1 so we know we don't know the face norm yet
        
        # Go through each pixel and calculate its normal
        for y in range(height):
            for x in range(width):
                
                n_i = y - 1     # North pixel index
                s_i = y + 1     # South pixel index
                e_i = x + 1     # East pixel index
                w_i = x - 1     # West pixel index
                
                # Check to see if the pixel is on the north edge
                if (n_i < 0):
                    n_i = height - 1         # Then wrap to the bottom
            
                # Check to see if the pixel is on the south edge
                if (s_i > height-1):
                    s_i = 0         # Then wrap to the top
                    
                # Check to see if the pixel is on the east edge
                if (e_i > width-1):
                    e_i = 0         # Then wrap to the left 
                    
                # Check to see if the pixel is on the west edge
                if (w_i < 0):
                    w_i = width - 1         # Then wrap to the right
                    
                
                # Get heights from each direction from the displ map
                n = displ_arr[n_i, x]
                s = displ_arr[s_i, x]
                e = displ_arr[y, e_i]
                w = displ_arr[y, w_i]
                
                #ew = normalize(np.array((2, 0, w-e)))  
                #ns = normalize(np.array((0, 2, n-s)))  
                #result = np.cross(ew, ns)
                
                #result = normalize(result)
                
                #print(result)
                
                result = np.array((2*(e-w), 2*(s-n), 4))
                result = self.normalize(result)
                
                #result = result * 0.5 + 0.5
                
                # Add results to normal map array
                normal[y, x] = result
                #print(normal[y, x])
        
        #print(np.amax(normal))
        #print(np.amin(normal))
        
        normal = (normal * 255).astype(np.uint8)
        #print(normal)
        return normal
    
    
    
    ## Create a roughness and displacement map
    def roughAndDispl(self, gray_im):
        """
        Create roughness and displacement maps from material
        gray_im = grayscale material image
        Returns roughness map and displacement map
        """
        
        gray_im2 = (gray_im.copy()).astype(np.float64) / 255  # Convert to float
        
        #print(np.amax(gray_im2))
        #print(np.amin(gray_im2))
        
        rough_im = gray_im2 * 1.7                     # Create roughness map
        
        #print(np.amax(rough_im))
        #print(np.amin(rough_im))
        
        rough_im = np.clip(rough_im, 0, 1)            # Clip to [0, 1]
        rough_im = (rough_im * 255).astype(np.uint8)  # Convert to int
        
        displ_im = cv2.GaussianBlur(gray_im2, (0, 0), 1)  # Create displ map
        displ_im = (displ_im * 255).astype(np.uint8)      # Convert to int
        
        return rough_im, displ_im
    
    
    
    ## Create a random seamless texture
    def createRanSeamlessTex(self, sample_img_fn, w, h):
        """
        Create random seamless texture from a texture image.
        
        Inputs:
          sample_img_fn:  sample image file name as string
          w:              desired output width
          h:              desire output height
        """
    
        sample_img = cv2.cvtColor(cv2.imread(sample_img_fn), cv2.COLOR_BGR2RGB)
        plt.imshow(sample_img)
        plt.tick_params(left = False, right = False , labelleft = False ,
         labelbottom = False, bottom = False)
        plt.title('Concrete Sample')
        plt.show()
    
        
        ## Random Texture
        out_size_w = w     # Output width
        patch_size_w = 25  # Patch width
        overlap_w = 12     # Overlap width
    
        remainder_w = (out_size_w  - patch_size_w) % (patch_size_w - overlap_w)  # Want remainder to be zero for no border
        out_size_w = out_size_w + (patch_size_w - overlap_w) - remainder_w       # Update output width for zero border
    
        out_size_h = h     # Output height
        patch_size_h = 25  # Patch height
        overlap_h = 12     # Overlap height
    
        remainder_h = (out_size_h  - patch_size_h) % (patch_size_h - overlap_h)  # Want remainder to be zero for no border
        out_size_h = out_size_h + (patch_size_h - overlap_h) - remainder_h       # Update output height for zero border
    
        tol = 2
        conc_mat = self.quilt_cut(sample_img, out_size_w, out_size_h, patch_size_w, patch_size_h, overlap_w, overlap_h, tol)
    
        conc_mat = conc_mat[0:h, 0:w, :]  # Trim concrete material so it's desired size
    
        #print(cropped.shape)
        #print(conc_mat.shape)
        
        plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)
        plt.title('Seam Finding Texture')
        plt.imshow(conc_mat)
        
        conc_mat2 = conc_mat.copy()
        conc_mat2[:, :, 0] = conc_mat[:, :, 2]
        conc_mat2[:, :, 2] = conc_mat[:, :, 0]
        cv2.imwrite("textures/conc_w_seams.jpg", conc_mat2)
        
        
        ## Seamless
        output = np.zeros(conc_mat.shape)
        conc_mat_fl = conc_mat.copy().astype('double') / 255.0  # Convert conc_mat to float
        for b in np.arange(3):
            output[:,:,b] = self.seamless(conc_mat_fl[:,:,b].copy())
    
        plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)
        plt.title('Seamless Final Texture')
        plt.imshow(output)
        
        # Convert output to uint8
        output2 = np.clip(output, 0, 1)
        output2 =(output2 * 255).astype(np.uint8)
        
        # Switch from rgb to bgr, so it saves correctly
        output3 = output2.copy()
        output3[:, :, 0] = output2[:, :, 2]
        output3[:, :, 2] = output2[:, :, 0]
        cv2.imwrite("textures/conc_rgb.jpg", output3)
        
        
        ## Create normal map
        output_gray = cv2.cvtColor(output2, cv2.COLOR_BGR2GRAY)
        norm_arr = self.createNormalMap(output_gray)
        
        # Switch from rgb to bgr, so it saves correctly
        norm_arr2 = norm_arr.copy()
        norm_arr2[:, :, 0] = norm_arr[:, :, 2]
        norm_arr2[:, :, 2] = norm_arr[:, :, 0]
        cv2.imwrite("textures/conc_norm.jpg", norm_arr2)
        
        plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)
        plt.title('Normal Texture')
        plt.imshow(norm_arr)
        
        
        ## Roughness and Displacement
        rough_arr, displ_arr = self.roughAndDispl(output_gray)
        cv2.imwrite("textures/conc_rough.jpg", rough_arr)
        cv2.imwrite("textures/conc_displ.jpg", displ_arr)
        
        plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)
        plt.title('Roughness Texture')
        plt.imshow(rough_arr, cmap='gray')
        
        plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)
        plt.title('Displacement Texture')
        plt.imshow(displ_arr, cmap='gray')
        
    
        return output2, norm_arr, rough_arr, displ_arr
    
    
    
    




    
    
    



