function ppower = POWERPOLY(p,n)
    ppower = p;
    i = 1;
    while i < n
        ppower = conv(ppower,p);
        i = i + 1;
    end
end