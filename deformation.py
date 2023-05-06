import pandas as pd
from cell_location import cellresults
from tip_micropipette import pipetteresults
from tip_micropipette import pipetteresults2
import re


# Convert the cell data and micropipette data to data frames
celllocation_df = pd.DataFrame(cellresults, columns=['file_name','major_axis','minor_axis','width','height','centerX','centerY'])
micropipetteleft_df = pd.DataFrame(pipetteresults, columns=['file_name', 'leftmost_pixel_x', 'leftmost_pixel_y'])
micropipetteright_df = pd.DataFrame(pipetteresults2, columns=['file_name', 'rightmost_pixel_x', 'rightmost_pixel_y'])

# Merge the data frames
merged_df = pd.merge(celllocation_df, micropipetteleft_df, on='file_name')
merged_df = pd.merge(merged_df, micropipetteright_df, on='file_name')

merged_df['radius'] = merged_df['minor_axis'].iloc[0] / 2
merged_df['distance'] = ((merged_df['leftmost_pixel_x'] - merged_df['centerX'])**2 + (merged_df['leftmost_pixel_y'] - merged_df['centerY'])**2)**0.5

print(merged_df['radius'])
print(merged_df['distance'])

deformation = merged_df['distance'] - merged_df['radius']

for index, row in merged_df.iterrows():
    deformation = row['distance'] - row['radius']
    if deformation > 2:
        print( "there is no deformation : ", row['file_name'], deformation)
        merged_df.loc[index, 'no deformation'] = deformation
        merged_df.loc[index, 'deformation'] = False
        merged_df.loc[index, 'contact'] = False
    elif deformation < 0:
        print("there is deformation : ", row['file_name'], deformation)
        merged_df.loc[index, 'no deformation'] = False
        merged_df.loc[index, 'deformation'] = deformation
        merged_df.loc[index, 'contact'] = False
    elif 0 <= deformation <= 2:
        print("there is a contact : ", row['file_name'], deformation)
        merged_df.loc[index, 'no deformation'] = False
        merged_df.loc[index, 'deformation'] = False
        merged_df.loc[index, 'contact'] = deformation

#sort file name
merged_df['frame_number'] = merged_df['file_name'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
merged_df = merged_df.sort_values(by=['frame_number'])

# Save the merged data frame with the contact column
merged_df.to_excel('deformation.xlsx', index=False)

#######################################################################################

merged_df['radiusperFrame'] = merged_df['minor_axis']/ 2
merged_df['distance'] = ((merged_df['leftmost_pixel_x'] - merged_df['centerX'])**2 + (merged_df['leftmost_pixel_y'] - merged_df['centerY'])**2)**0.5

print(merged_df['radiusperFrame'])
print(merged_df['distance'])

deformation_perFrame = merged_df['distance'] - merged_df['radiusperFrame']

for index, row in merged_df.iterrows():
    deformation_perFrame = row['distance'] - row['radiusperFrame']
    if deformation_perFrame > 2:
        print( "there is no deformation : ", row['file_name'], deformation_perFrame)
        merged_df.loc[index, 'no deformation_perFrame'] = deformation_perFrame
        merged_df.loc[index, 'deformation_perFrame'] = False
        merged_df.loc[index, 'contact_perFrame'] = False
    elif deformation_perFrame < 0:
        print("there is deformation : ", row['file_name'], deformation_perFrame)
        merged_df.loc[index, 'no deformation_perFrame'] = False
        merged_df.loc[index, 'deformation_perFrame'] = deformation_perFrame
        merged_df.loc[index, 'contact_perFrame'] = False
    elif 0 <= deformation_perFrame <= 2:
        print("there is a contact : ", row['file_name'], deformation_perFrame)
        merged_df.loc[index, 'no deformation'] = False
        merged_df.loc[index, 'deformation'] = False
        merged_df.loc[index, 'contact_perFrame'] = deformation_perFrame

#sort file name
merged_df['frame_number'] = merged_df['file_name'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
merged_df = merged_df.sort_values(by=['frame_number'])

# Save the merged data frame with the contact column
merged_df.to_excel('deformation.xlsx', index=False)

#########################################################################################
