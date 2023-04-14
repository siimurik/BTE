import os
from tqdm import tqdm

def convertGXStoCSV():
    # structure filesGXS contains information about all GXS files in the current directory
    filesGXS = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.GXS')]

    # loop over all GXS files
    for gxs_file in filesGXS:
        # define the name of the file without extension
        nameOnly = os.path.splitext(gxs_file)[0]

        # if the corresponding CSV file does not exist
        if not os.path.isfile(nameOnly + '.CSV'):
            print(f'Convert {gxs_file} to {nameOnly}.CSV: ', end='')

            # open GXS file for reading
            with open(gxs_file, 'r') as fdGXS:
                # open CSV file for writing
                with open(nameOnly + '.CSV', 'w') as fdCSV:
                    # get the size of the file
                    fileSize = os.path.getsize(gxs_file)

                    # create a tqdm progress bar
                    progress = tqdm(total=fileSize, unit='B', unit_scale=True)

                    # loop through each line in the GXS file
                    for line in fdGXS:
                        ii = 0
                        for iii in range(6):
                            if line[ii+8] == '+' or line[ii+8] == '-' and line[ii+7] != ' ':
                                # insert "E" (1.0+01 --> 1.0E+01)
                                str1 = line[ii:ii+7] + 'E' + line[ii+8:ii+10]

                            elif line[ii+9] == '+' or line[ii+9] == '-' and line[ii+8] != ' ':
                                # insert "E" (1.0+01 --> 1.0E+01)
                                str1 = line[ii:ii+8] + 'E' + line[ii+9:ii+11]

                            else:
                                str1 = ' ' + line[ii:ii+10+1]

                            # write the line inserting semicolons
                            fdCSV.write(f' {str1};')
                            ii += 11

                        fdCSV.write(line[67:70] + ';')
                        fdCSV.write(line[71:72] + ';')
                        fdCSV.write(line[73-1:75] + ';')
                        fdCSV.write(line[76:80] + '\n')

                        # update the tqdm progress bar
                        progress.update(len(line))

                    # close the tqdm progress bar
                    progress.close()

            print('Done')

if __name__ == '__main__':
    convertGXStoCSV()
