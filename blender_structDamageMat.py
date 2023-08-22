# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Casey Rodgers

Create Structural Damage Material Textures for Blender
    
References: 
// Structural Damage
    -“How to Add a New Stop to the Color Ramp?” Blender Stack Exchange, 1 Sept. 2020 https://blender.stackexchange.com/questions/189712/how-to-add-a-new-stop-to-the-color-ramp. 
    -“Make a Photo-Realistic Concrete Material with Cracks in Blender 2.82.” YouTube, YouTube, 6 Mar. 2020, https://www.youtube.com/watch?v=8Odon-JrQ7o. 
    -“Shadernode(NodeInternal).” ShaderNode(NodeInternal) - Blender Python API, 17 Dec. 2021, https://docs.blender.org/api/current/bpy.types.ShaderNode.html. 
    - https://blender.stackexchange.com/questions/23436/control-cycles-eevee-material-nodes-and-material-properties-using-python

"""

## Imports
import bpy



class structMatDamageBlender():

    """ Update Color Ramp """
    # Update color ramp by adding colors at specific positions
    # colorRamp = color ramp object
    # colors = array of colors wanted (one 4-tuple per stop)
    # positions = array of positions wanted (one float per stop)
    def updateColorRamp(self, colorRamp, colors, positions):
    
        # Go through each stop we want
        for i in range(len(colors)):
            
            # Use existing stop or create a new one
            if (i == 0 or i == 1):
                colorRamp.elements[i].position = positions[i]  # Set position of existing stop
            else:
                colorRamp.elements.new(positions[i])    # Create new stop
            
            colorRamp.elements[i].color = colors[i]     # Set color
            
          
      
    
    ############################################
    ############################################
    """ Create material with textures in a folder """
    # Creates damaged blender material from image textures
    # texFolder = folder with the textures
    # strImgName = inputted structure damage textures folder
    # matImgName = inputted material textures folder
    
    def createDamageMaterial(self, texFolder, inImgName):
        
        """ Create Material """
        
        # Damage Name
        folder = texFolder + '\\' + inImgName                 # Folder
        mat_name =  inImgName                 # Material name
        
            
        # Check if material exists and create new one if not
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]      # Get existing material
        else:
            mat = bpy.data.materials.new(mat_name)     # Create material
        
        mat.use_nodes = True            # Use nodes
        nodes = mat.node_tree.nodes     # Set material nodes to name nodes
        nodes.clear()                   # Clear all nodes to start clean
        links = mat.node_tree.links     # Set material links to name links
        
        space_w = 100              # Amount of width space between nodes
        space_h = 100*2              # Amount of height space between nodes
        randScale = 300             # Random scaling variable
        
        
        
        ############################################
        """ RGB """
        
        curr_x1 = 0                  # Current width (for location)
        curr_y1 = 0                 # Current height (for location)
        
        
        ## Create Image Texture Node 1
        # Properties
        node_imgTex1 = nodes.new(type='ShaderNodeTexImage')   # Create node
        node_imgTex1.extension = 'REPEAT'         # Image extension
        node_imgTex1.image = bpy.data.images.load(folder + '-rgbImage.jpg')         # Image
        node_imgTex1.interpolation = 'Linear'         # Image interpolation
        node_imgTex1.projection = 'FLAT'         # Image projection
        node_imgTex1.location = curr_x1, curr_y1                # Location
        curr_x1 = curr_x1 + node_imgTex1.width + space_w       # Update current x value
        curr_y1 = curr_y1 - 100       # Update current y value
        
        
        
        ############################################
        """ Metallic """
        
        curr_x2 = 0                  # Current width (for location)
        curr_y2 = curr_y1 - space_h    # Update current height (for location)
        
        
        ## Create Image Texture Node 2
        # Properties
        node_imgTex2 = nodes.new(type='ShaderNodeTexImage')   # Create node
        node_imgTex2.extension = 'REPEAT'         # Image extension
        node_imgTex2.image = bpy.data.images.load(folder + '-metallic.jpg')         # Image
        node_imgTex2.interpolation = 'Linear'         # Image interpolation
        node_imgTex2.projection = 'FLAT'         # Image projection
        node_imgTex2.location = curr_x2, curr_y2                # Location
        curr_x2 = curr_x2 + node_imgTex2.width + space_w       # Update current x value
        curr_y2 = curr_y2 - 100                 # Update current y value
        
        
        
        ############################################
        """ Bump """
        
        curr_x3 = 0                  # Current width (for location)
        curr_x32 = curr_x3          # Current width (for second branch)
        curr_x33 = curr_x3          # Current width (for third branch)
        curr_y3 = curr_y2 - space_h    # Update current height (for location)
        curr_y32 = curr_y3          # Current height for second branch
        curr_y33 = curr_y3          # Current width (for third branch)
        
        
        
        ## Create Image Texture Node 3 (Mat bump)
        # Properties
        node_imgTex3 = nodes.new(type='ShaderNodeTexImage')   # Create node
        node_imgTex3.extension = 'REPEAT'         # Image extension
        node_imgTex3.image = bpy.data.images.load(folder + '-mat_bump.jpg')         # Image
        node_imgTex3.interpolation = 'Linear'         # Image interpolation
        node_imgTex3.projection = 'FLAT'         # Image projection
        node_imgTex3.location = curr_x3, curr_y3                # Location
        curr_x3 = curr_x3 + node_imgTex3.width + space_w       # Update current x value
        curr_y32 = curr_y32 - node_imgTex3.height - space_h    # Update y location for second branch
        curr_y33 = curr_y33 - node_imgTex3.height - space_h    # Update y location for 3rd branch
        
        
        ## Create Image Texture Node 4 (spall bump)
        # Properties
        node_imgTex4 = nodes.new(type='ShaderNodeTexImage')   # Create node
        node_imgTex4.extension = 'REPEAT'         # Image extension
        node_imgTex4.image = bpy.data.images.load(folder + '-spall_bump.jpg')         # Image
        node_imgTex4.interpolation = 'Linear'         # Image interpolation
        node_imgTex4.projection = 'FLAT'         # Image projection
        node_imgTex4.location = curr_x32, curr_y32                # Location
        curr_x32 = curr_x32 + node_imgTex4.width + space_w       # Update current x value
        curr_y33 = curr_y33 - node_imgTex4.height - space_h    # Update y location for 3rd branch

        
        ## Create Image Texture Node 5 (Cracks Displ)
        # Properties
        node_imgTex5 = nodes.new(type='ShaderNodeTexImage')   # Create node
        node_imgTex5.extension = 'REPEAT'         # Image extension
        node_imgTex5.image = bpy.data.images.load(folder + '-cracks-displ.jpg')         # Image
        node_imgTex5.interpolation = 'Linear'         # Image interpolation
        node_imgTex5.projection = 'FLAT'         # Image projection
        node_imgTex5.location = curr_x33, curr_y33                # Location
        curr_x33 = curr_x33 + node_imgTex5.width + space_w       # Update current x value
        
        
        ## Create Math Node 1 (Mat bump)
        # Properties
        node_math1 = nodes.new(type='ShaderNodeMath')   # Create node
        node_math1.operation = 'MULTIPLY'         # Multiply
        node_math1.use_clamp = False         # Use clamp
        node_math1.location = curr_x3, curr_y3                # Location
        curr_x3 = curr_x3 + node_math1.width + space_w       # Update current x value
        
        # Inputs
        #print(node_vor.inputs.keys())
        node_math1.inputs[1].default_value = 3.00           # Value
        link1_math1 = links.new(node_imgTex3.outputs['Color'], node_math1.inputs[0])  # Vector
        
        
        ## Create Math Node 2 (Spall bump)
        # Properties
        node_math2 = nodes.new(type='ShaderNodeMath')   # Create node
        node_math2.operation = 'MULTIPLY'         # Multiply
        node_math2.use_clamp = False         # Use clamp
        node_math2.location = curr_x32, curr_y32                # Location
        curr_x32 = curr_x32 + node_math2.width + space_w       # Update current x value
        
        # Inputs
        node_math2.inputs[1].default_value = 18.00           # Value
        link1_math2 = links.new(node_imgTex4.outputs['Color'], node_math2.inputs[0])  # Vector
        
        
        ## Create Math Node 3 (Cracks Displ)
        # Properties
        node_math3 = nodes.new(type='ShaderNodeMath')   # Create node
        node_math3.operation = 'MULTIPLY'         # Multiply
        node_math3.use_clamp = False         # Use clamp
        node_math3.location = curr_x33, curr_y33                # Location
        curr_x33 = curr_x33 + node_math3.width + space_w       # Update current x value
        
        # Inputs
        node_math3.inputs[1].default_value = -1.0           # Value
        link1_math3 = links.new(node_imgTex5.outputs['Color'], node_math3.inputs[0])  # Vector
        
        
        ## Create Math Node 4 (Combine mat and spall bump)
        # Properties
        node_math4 = nodes.new(type='ShaderNodeMath')   # Create node
        node_math4.operation = 'ADD'         # Add
        node_math4.use_clamp = False         # Use clamp
        node_math4.location = curr_x3, curr_y3                # Location
        curr_x3 = curr_x3 + node_math4.width + space_w       # Update current x value
        
        # Inputs
        link1_math4 = links.new(node_math1.outputs['Value'], node_math4.inputs[0]) 
        link2_math4 = links.new(node_math2.outputs['Value'], node_math4.inputs[1]) 
        
        
        ## Create Math Node 5 (bump and displ)
        # Properties
        node_math5 = nodes.new(type='ShaderNodeMath')   # Create node
        node_math5.operation = 'ADD'         # Add
        node_math5.use_clamp = False         # Use clamp
        node_math5.location = curr_x3, curr_y3                # Location
        curr_x3 = curr_x3 + node_math5.width + space_w       # Update current x value
        
        # Inputs
        link1_math5 = links.new(node_math3.outputs['Value'], node_math5.inputs[1]) 
        link2_math5 = links.new(node_math4.outputs['Value'], node_math5.inputs[0]) 
        
        
        ## Create Bump Node 1
        # Properties
        node_bump1 = nodes.new(type='ShaderNodeBump')     # Create node
        node_bump1.invert = True                      # Invert bump map dir?
        node_bump1.location = curr_x3, curr_y3
        curr_x3 = curr_x3 + node_bump1.width + space_w       # Update current x value
        
        # Inputs
        #print(node_bump1.inputs.keys())
        node_bump1.inputs['Strength'].default_value = 1.0        # Strength
        node_bump1.inputs['Distance'].default_value = 1.0        # Distance
        link1_bump1 = links.new(node_math5.outputs['Value'], node_bump1.inputs['Height'])  # Height
        
        
        
        ############################################
        """ Displacement """
        
        curr_x4 = 0              # Reset current x
        curr_y4 = curr_y33 - node_imgTex5.height - space_h      # Update current y value
        
        
        ## Create Image Texture Node 6 (Displacement)
        # Properties
        node_imgTex6 = nodes.new(type='ShaderNodeTexImage')   # Create node
        node_imgTex6.extension = 'REPEAT'         # Image extension
        node_imgTex6.image = bpy.data.images.load(folder + '-displacement.jpg')         # Image
        node_imgTex6.interpolation = 'Linear'         # Image interpolation
        node_imgTex6.projection = 'FLAT'         # Image projection
        node_imgTex6.location = curr_x4, curr_y4                # Location
        curr_x4 = curr_x4 + node_imgTex6.width + space_w       # Update current x value
        
        
        ## Create Math Node 6
        # Properties
        node_math6 = nodes.new(type='ShaderNodeMath')   # Create node
        node_math6.operation = 'ADD'         # Add
        node_math6.use_clamp = False         # Use clamp
        node_math6.location = curr_x4, curr_y4               # Location
        curr_x4 = curr_x4 + node_math6.width + space_w       # Update current x value
        
        # Inputs
        node_math6.inputs[1].default_value = -0.4           # Value
        link1_math6 = links.new(node_imgTex6.outputs['Color'], node_math6.inputs[0])  # Vector
        
        
        ## Create Displacement Node
        # Properties
        node_displ1 = nodes.new(type='ShaderNodeDisplacement')   # Create node
        node_displ1.space = 'OBJECT'         # Space of the input height
        node_displ1.location = curr_x4, curr_y4               # Location
        curr_x4 = curr_x4 + node_displ1.width + space_w       # Update current x value
        
        # Inputs
        node_displ1.inputs['Midlevel'].default_value = 0.5           # Midlevel
        node_displ1.inputs['Scale'].default_value = 1.0           # Scale
        link1_displ1 = links.new(node_math6.outputs['Value'], node_displ1.inputs['Height'])  # Vector
        
    
    
        ############################################
        """ Add everything all together to Principled BDSF and Material Output"""
        
        curr_x5 = max(curr_x1, curr_x2, curr_x3, curr_x32, curr_x33, curr_x4) + space_w   # Find max width to start and then add a space
        curr_y5 = 0    # Y location
        
        
        ## Create Principled BSDF Node
        # Properties
        node_bsdf1 = nodes.new(type='ShaderNodeBsdfPrincipled')     # Create node
        node_bsdf1.distribution = 'GGX'                       # Distribution
        node_bsdf1.subsurface_method = 'BURLEY'               # Subsurface method
        node_bsdf1.location = curr_x5, curr_y5         # Location
        curr_x5 = curr_x5 + node_bsdf1.width + space_w       # Update current x value
        
        # Inputs
        link1_bsdf1 = links.new(node_imgTex1.outputs['Color'], node_bsdf1.inputs['Base Color'])  # Base color
        link2_bsdf1 = links.new(node_imgTex2.outputs['Color'], node_bsdf1.inputs['Metallic'])  # Metallic
        link3_bsdf1 = links.new(node_bump1.outputs['Normal'], node_bsdf1.inputs['Normal'])  # Normal
        
        
        ## Create Material Output Node
        # Properties
        node_matOut1 = nodes.new(type='ShaderNodeOutputMaterial')     # Create node
        node_matOut1.is_active_output = True                       # Is active output?
        node_matOut1.target = 'ALL'                       # Which renderer and viewport shading types to use the shaders for
        node_matOut1.location = curr_x5, curr_y3         # Location
        curr_x5 = curr_x5 + node_matOut1.width + space_w       # Update current x value
        
        # Inputs
        #print(node_bump1.inputs.keys())
        link1_matOut1 = links.new(node_bsdf1.outputs['BSDF'], node_matOut1.inputs['Surface'])  # Surface
        link2_matOut1 = links.new(node_displ1.outputs['Displacement'], node_matOut1.inputs['Displacement'])  # Displacement
        


############################################
############################################
############################################
""" Main Script """

# Inputs
texFolder = 'textures'  # Folder with the damaged textures
inImgName = 'allFloors2' # Inputted image name (ex. floor1)

cd = structMatDamageBlender()
cd.createDamageMaterial(texFolder, inImgName)
