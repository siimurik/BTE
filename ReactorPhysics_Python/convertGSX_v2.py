import os
from tqdm import tqdm

def convertGXStoCSV():
    # structure filesGXS contains information about all GXS files in the current directory
    filesGXS = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.GXS')]
    # Better explanation:
    #--------------------
    # os.listdir('.') lists all files and directories in the current directory ('.' refers to the 
    # current directory).
    # The if statement filters the results to include only regular files that end with ".GXS".
    # The list comprehension [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.GXS')] 
    # creates a new list containing the filtered file names.
    # So the result is a list of strings, where each string represents the name of a file in the 
    # current directory that ends with ".GXS".

    # loop over all GXS files
    for gxs_file in filesGXS:
        # define the name of the file without extension
        nameOnly = os.path.splitext(gxs_file)[0]
        # Essentialy what this means:
        #----------------------------
        # For each file in the filesGXS list, it defines the name of the file without
        # its extension (i.e., removes the ".GXS" extension) and assigns it to the nameOnly variable.
        # This is useful for creating the corresponding CSV file name, which will have the same name
        # as the GXS file, but with a ".CSV" extension instead. For example, if the GXS file is 
        # named "file1.GXS", then the corresponding CSV file will be named "file1.CSV".

        # if the corresponding CSV file does not exist
        if not os.path.isfile(nameOnly + '.CSV'):
            print(f'Convert {gxs_file} to {nameOnly}.CSV: ', end='')
            # Bit more clearly:
            #-------------------
            # If such a file does not exist, the script prints a message saying it is 
            # going to convert the GXS file to a CSV file with the same name as the GXS 
            # file, and sets the end parameter of the print function to an empty string 
            # to prevent the print function from adding a newline character.
            # NB! If a file with a similar name already exist, the code won't do anything
            # so it is strongly advised that the old CSV file is deleted before rerunning
            # the code.

            # open GXS file for reading
            with open(gxs_file, 'r') as fdGXS:
                # open CSV file for writing
                with open(nameOnly + '.CSV', 'w') as fdCSV:
                    # get the size of the file
                    # necessary for the progress bar to work
                    fileSize = os.path.getsize(gxs_file)

                    # create a tqdm progress bar
                    progress = tqdm(total=fileSize, unit='B', unit_scale=True)

                    # loop through each line in the GXS file
                    for line in fdGXS:
                        ii = 0
                        for iii in range(6):
                            if line[ii+8] == '+' or line[ii+8] == '-' and line[ii+7] != ' ':
                                # insert "E" (1.0+01 --> 1.0E+01)
                                str1 = line[ii:ii+7+1] + 'E' + line[ii+8:ii+10+1]
                                                # "+1" here bc in here"8=7+1"
                                                # if you put "7" like in MATLAB
                                                # then Python goes to "6" instead 
                                                # of "7" bc starting from 0 is so fun... ಠ_ಠ
                            elif line[ii+9] == '+' or line[ii+9] == '-' and line[ii+8] != ' ':
                                # insert "E" (1.0+01 --> 1.0E+01)
                                str1 = line[ii:ii+8+1] + 'E' + line[ii+9:ii+11]
                                                # same story here
                                                # 9 here needs to be ↑ 9 here
                            else:
                                str1 = ' ' + line[ii:ii+10+1]
                                                        # +1 here bc otherwise numbers like "-1" 
                                                        # will just be "-" instead
                            # write the line inserting semicolons
                            fdCSV.write(f' {str1};')
                            ii += 11    # Fucks up the first number in columns 3-7 if increased
                                        # 1.000000E-5 ;  .000000-4 5;  000000-4 7.;  00000-4 1.0;  0000-3
                        fdCSV.write(line[67-1:70] + ';') # needs "-1" bc looks prettier; adds space to the left of the number
                        fdCSV.write(line[71-1:72] + ';') # needs "-1" bc looks prettier; adds space to the left of the number
                        fdCSV.write(line[73-1:75] + ';') # "-1" bc otherwise "51" instead of "451" in second to last column
                        fdCSV.write(line[76-1:80] + '\n')# needs "-1" bc looks prettier; adds space to the left of the number
                                                         # also bc "diff" command in terminal gets angry otherwise
                        # update the tqdm progress bar
                        progress.update(len(line)+1) # "+1" here bc otherwise progress bar
                                                     # only goes to 99%, bc starting numbers
                                                     # from 0 definitely won't cause problems
                    # close the tqdm progress bar
                    progress.close()

            print('Done')

# Theoretically no need to do it this way.
# If you are a C supremist you'll prolly 
# enjoy this more. Either way, this function
# can be now easily copied or run separately 
# as shown here below.
if __name__ == '__main__':
    convertGXStoCSV()
