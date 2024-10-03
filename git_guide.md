## M4 code pushing guide, do read the other notes below if u do wanna find out
git status (to check where u are and to check what files are updated)
git add -A ':!venv2'  (stage all files to push to repo except for venv2 folder)
(git status again to make sure)
git commit -m '<commit_message>'
git push origin <branchname>  (sandra pushes this to origin main)



# Git set up
1. check version: git --version

# Cloning Repositories

1. fork repo
2. click <code>, copy https link 
3. in vsc, make/enter desired directory to put repo in
4. terminal: git clone <link> 
5. enter the repo, cd <repo_name>
6. ls gives all files in the repo
7. make sure youre in the right branch, right directory: git status

// if you forked a main branch
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean

// origin is the name of the remote repo

# Creating and coding in a branch 

1. git branch <branch_name> 
2. check if branch successfully created: git branch 
3. git checkout <branch_name>           ------ dont forget this step!!! 
4. git status to ensure                 ------ this is very important!!! 
5. go ham with the code changes, just make sure u stay on the correct branch

**git won't let you change to a branch when you have unsaved changes in the current branch within the local repo
**the local repo is a hidden .git file 


**refer to the block diagram in the lesson for path 
# Saving changes 

1. when changes have been made, 'upload' code to staging area: git add . 
    - the . just indicates 'all'
    - vsc highlights file modified in blue
2. can check for the changes made in git status, check if its in the staging area
3. commit the changes to update the local repo: git commit -m "<commit_message>"

--- if you want to add all but some directories:
git add -A ':!<directory/file_to_exclude>'   




# uploading changes to the remote repo
1. after staging and committing, push to remote repo: git push origin <branch_name> 
2. might ask for the password, lol
3. changes should be visible in the remote repo


# merging branches
1. nightmare, dont do in vsc do it in the repo instead


