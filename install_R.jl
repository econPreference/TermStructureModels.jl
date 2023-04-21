##When install RCall##
# 1. Install R form internet
# 2. In R, run " R.home() " and copy the home address
# 3. In R, run " install.packages("GIGrvg") " and " install.packages("glasso") " to install the packages
# 4. In Juila, run  " ENV["R_HOME"]="...the address in step 2..." "
# 5. In Juila, run " using Pkg " and " Pkg.add("RCall") "
using Conda, Pkg
# ENV["R_HOME"] = "/Library/Frameworks/R.framework/Resources" # For my mac computer
ENV["R_HOME"] = "C:/Program Files/R/R-4.2.3" # For my windows computer
Pkg.build("RCall")
using RCall
R"install.packages('MASS', repos='https://cloud.r-project.org/', type = 'binary')"
R"install.packages('qgraph', repos='https://cloud.r-project.org/', type = 'binary')"
R"install.packages('GIGrvg', repos='https://cloud.r-project.org/', type = 'binary')"
println("All R packages are installed.")
######################