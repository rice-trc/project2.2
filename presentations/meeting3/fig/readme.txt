mogrify -format png *.eps

If it doesn't work, the policy file for ImageMagic should be changed. To
eliminate all usage restriction, just do

sudo mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xmlout

Which is reverted by renaming it back to the original name
sudo mv /etc/ImageMagick-6/policy.xmlout /etc/ImageMagick-6/policy.xml

See
https://askubuntu.com/a/1081907

epstopdf
sudo apt install texlive-font-utils

for i in *.eps; do epstopdf "$i"; done

or multiline
for f in "/path/to/image/folder with spaces/"*.eps; do
    epstopdf "$f"
done
