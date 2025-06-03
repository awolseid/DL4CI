list patkey cummonth dead cens mtxspan jc jc_nd jc_0 in 2035/2045
mkspline spline = cummonth, cubic knots( 8        18        37        61        83)
xi: logistic mtxspan onprd2 i.dmrd i.haq i.gs i.esrc i.jc i.smokenow onprd2_nd i.dmrd_nd i.haq_nd i.gs_nd i.esrc_nd i.jc_nd onprd2_0 i.duration_0 i.age_0 i.year_0 i.dmrd_0 i.haq_0 i.gs_0 i.esrc_0 i.jc_0 i.smoke_0 sex i.edu_0 rapos cummonth spline* if cummonth<=mtx1stcu | mtx1stcu==.
predict pmtx if e(sample)
replace pmtx=1 if cummonth>mtx1stcu
replace pmtx=pmtx*mtxspan+(1-pmtx)*(1-mtxspan)
sort patkey cummonth
by patkey: replace pmtx=pmtx*pmtx[_n-1] if _n!=1
rename pmtx mtxdenom
summ mtxdenom, detail
xi: logistic mtxspan onprd2_0 i.duration_0 i.age_0 i.year_0 i.dmrd_0 i.haq_0 i.gs_0 i.esrc_0 i.jc_0 i.smoke_0 sex i.edu_0 rapos cummonth spline* if cummonth<=mtx1stcu | mtx1stcu==.
predict pmtx if e(sample)
replace pmtx=1 if cummonth>mtx1stcu
replace pmtx=pmtx*mtxspan+(1-pmtx)*(1-mtxspan)
sort patkey cummonth
by patkey: replace pmtx=pmtx*pmtx[_n-1] if _n!=1
rename pmtx mtxnum
gen stabweightmtx=mtxnum/mtxdenom
summ stabweightmtx, detail

xi: logistic cens mtxspan onprd2 i.dmrd i.haq i.gs i.esrc i.jc i.smokenow onprd2_nd i.dmrd_nd i.haq_nd i.gs_nd i.esrc_nd i.jc_nd onprd2_0 i.duration_0 i.age_0 i.year_0 i.dmrd_0 i.haq_0 i.gs_0 i.esrc_0 i.jc_0 i.smoke_0 sex i.edu_0 rapos cummonth spline*
predict pcens if e(sample)
replace pcens=1-pcens

sort patkey cummonth
by patkey: replace pcens=pcens*pcens[_n-1] if _n!=1
rename pcens censdenom
summ censdenom, detail
xi: logistic mtxspan onprd2_0 i.duration_0 i.age_0 i.year_0 i.dmrd_0 i.haq_0 i.gs_0 i.esrc_0 i.jc_0 i.smoke_0 sex i.edu_0 rapos cummonth spline*
predict pcens if e(sample)
replace pcens=1-pcens
sort patkey cummonth
by patkey: replace pcens=pcens*pcens[_n-1] if _n!=1
rename pcens censnum
summ censnum, detail
gen censweight=censnum/censdenom
summ censweight, detail
gen stabweightcens=stabweightmtx*censweight

xi: logistic dead mtxspan onprd2_0 i.duration_0 i.age_0 i.year_0 i.dmrd_0 i.haq_0 i.gs_0 i.esrc_0 i.jc_0 i.smoke_0 sex i.edu_0 rapos onprd2_nd i.dmrd_nd i.haq_nd i.gs_nd i.esrc_nd i.jc_nd onprd2 i.dmrd i.haq i.gs i.esrc i.jc i.smokenow cummonth spline*

xi: logistic dead mtxspan onprd2_0 i.duration_0 i.age_0 i.year_0 i.dmrd_0 i.haq_0 i.gs_0 i.esrc_0 i.jc_0 i.smoke_0 sex i.edu_0 rapos onprd2_nd i.dmrd_nd i.haq_nd i.gs_nd i.esrc_nd i.jc_nd onprd2 i.dmrd i.haq i.gs i.esrc i.jc i.smokenow cummonth spline* [pw=stabweightcens], cluster(patkey)








