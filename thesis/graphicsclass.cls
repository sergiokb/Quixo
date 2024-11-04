%% With modifications from file `copernicus.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% copernicus.dtx  (with options: `class')
%% 
%% -----------------------------------------------------------------
%% Author:     copernicus.org and le-tex publishing services
%% 
%% This file is part of the copernicus package for papers
%% published by Copernicus Publications (Copernicus GmbH).
%% 
%%       Copyright (C) 2020 by Copernicus Publications
%% -----------------------------------------------------------------

\ProvidesClass{graphicsclass}[2019/7/20 v0.1 McGill University Thesis Class]

\LoadClass[12pt]{report}[2000/05/19 v1.4b Standard LaTeX document class]
% Alter some LaTeX defaults for better treatment of figures:
    % See p.105 of "TeX Unbound" for suggested values.
    % See pp. 199-200 of Lamport's "LaTeX" book for details.
    %   General parameters, for ALL pages:
    \renewcommand{\topfraction}{0.9}	% max fraction of floats at top
    \renewcommand{\bottomfraction}{0.8}	% max fraction of floats at bottom
    %   Parameters for TEXT pages (not float pages):
    \setcounter{topnumber}{2}
    \setcounter{bottomnumber}{2}
    \setcounter{totalnumber}{4}     % 2 may work better
    \setcounter{dbltopnumber}{2}    % for 2-column pages
    \renewcommand{\dbltopfraction}{0.9}	% fit big float above 2-col. text
    \renewcommand{\textfraction}{0.07}	% allow minimal text w. figs
    %   Parameters for FLOAT pages (not text pages):
    \renewcommand{\floatpagefraction}{0.7}	% require fuller float pages
	% N.B.: floatpagefraction MUST be less than topfraction !!
    \renewcommand{\dblfloatpagefraction}{0.7}	% require fuller float pages

	% remember to use [htp] or [htpb] for placement
\newif\if@stage@final  \@stage@finaltrue

\newcommand{\change}[1]{{\color{black} #1}} 
\renewcommand{\contentsname}{Table of Contents}
\RequirePackage{array}

\DeclareRobustCommand*\chem[1]
 {\ensuremath{%
   {\mathcode`\-="0200\mathcode`\=="003D% no space around "-" and "="
    \ifx\f@series\testbx\mathbf{#1}\else\mathrm{#1}\fi}}}
\@ifpackageloaded{hyperref}
 {\begingroup
    \toks0=\expandafter{\pdfstringdefPreHook}%
    \xdef\pdfstringdefPreHook{%
      \the\toks0 %
      \let\noexpand\hack\noexpand\@gobble
      \let\noexpand\chem\noexpand\@firstofone}%
  \endgroup}
 {\relax}
\DeclareRobustCommand*\unit[1]
 {\ensuremath{%
   {\thinmuskip3mu\relax
    \def\mu{\text{\textmu}}\def~{\,}%
    \ifx\f@series\testbx\mathbf{#1}\else\mathrm{#1}\fi}}}
%reaction
\@definecounter{reaction}
%\def\thereaction{R\@arabic\c@reaction}
\def\thereaction{\thechapter.R\@arabic\c@reaction}%

\let\c@org@eq\c@equation
\let\org@theeq\theequation
\def\setreaction{%
  \global\let\c@equation\c@reaction
  \global\let\theequation\thereaction}
\def\setequation{%
  \global\let\c@equation\c@org@eq
  \global\let\theequation\org@theeq}
\newenvironment{reaction}
 {\let\c@equation\c@reaction
  \let\theequation\thereaction
  \incr@eqnum
  \mathdisplay@push
  \st@rredfalse \global\@eqnswtrue
  \mathdisplay{equation}}
 {\endmathdisplay{equation}%
  \mathdisplay@pop
  \ignorespacesafterend}
  
\newenvironment{rxnarray}
 {\let\c@equation\c@reaction
  \let\theequation\thereaction
  \eqnarray}
 {\endeqnarray
  \let\c@equation\c@reaction
  \let\theequation\thereaction}
\if@stage@final\else
  \newcounter{parentreaction}
  \newenvironment{subreactions}
   {\refstepcounter{reaction}%
    \protected@edef\theparentreaction{\thereaction}%
    \setcounter{parentreaction}{\value{reaction}}%
    \setcounter{reaction}{0}%
    \def\thereaction{\theparentreaction\alph{reaction}}%
    \ignorespaces}
   {\setcounter{reaction}{\value{parentreaction}}%
    \ignorespacesafterend}
\fi
\newcommand\tophline{\hline\noalign{\vspace{1mm}}}
\newcommand\middlehline{\noalign{\vspace{1mm}}\hline\noalign{\vspace{1mm}}}
\newcommand\bottomhline{\noalign{\vspace{1mm}}\hline}
\newcommand\hhline{\noalign{\vspace{1mm}}\hline\noalign{\vspace{1mm}}}
%%
\newdimen\tabularwidth
\def\@tabular{%
  \leavevmode
  \setbox\z@
  \hbox \bgroup $\col@sep\tabcolsep \let\d@llarbegin\begingroup
                                    \let\d@llarend\endgroup
  \@tabarray}
\def\endtabular{\endarray $\egroup
  \global\tabularwidth\wd\z@
  \unhbox\z@}
\expandafter\let\csname endtabular*\endcsname=\endtabular
\DeclareRobustCommand\belowtable[1]{%
  \par\vspace{1ex}%
  {\if@stage@final\else\def\@tempa{table}\ifx\@captype\@tempa\else\centering\fi\fi
   \ifdim\tabularwidth<10mm\tabularwidth\hsize\fi
   \leavevmode\vrule\@width\z@\@height2ex\@depth\z@
   \parbox{\tabularwidth}{\raggedright\scriptsize#1}%
   \if@stage@final\else\par\fi}}
   
   
\@definecounter{listings}
\def\listingsname{Listing}
\def\setlistings{\global\let\c@figure\c@listings\global\let\figurename\listingsname}   
   
\providecommand\appendixname{Appendix}
\let\orisection=\section
\let\theolds=\thesection
\let\theoldss=\thesubsection
\def\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \def\thesection{\@Alph\c@section}%
  \def\thesubsection{\thesection\@arabic\c@subsection}%
  \setcounter{secnumdepth}{4}%
  \@addtoreset{equation}{section}%
  \setcounter{equation}{0}%
  \setcounter{reaction}{0}%
  \def\theequation{\thesection\@arabic\c@equation}%
  \def\thereaction{\thesection R\@arabic\c@reaction}%
  \let\c@org@lst\c@listings%
  \let\c@org@eq\c@equation%
  \let\org@theeq\theequation%
  \if@stage@final
    \@addtoreset{figure}{section}%
    \setcounter{figure}{0}%
    \def\thefigure{\thesection\@arabic\c@figure}%
    \@addtoreset{table}{section}%
    \setcounter{table}{0}%
    \def\thetable{\thesection\@arabic\c@table}%
    \@addtoreset{listings}{section}%
    \setcounter{listings}{0}%
    \def\thelistings{\thesection\@arabic\c@listings}%
  \fi
  \global\let\old@sect\@sect
  \def\@sect##1##2##3##4##5##6[##7]##8{%
    \ifnum ##2>\c@secnumdepth
      \let\@svsec\@empty
    \else
      \refstepcounter{##1}%
      \ifnum ##2=\@ne
        \def\@tempa{##8}%
        \protected@edef\@svsec{%
          \appendixname\space\csname the##1\endcsname
          \ifx\@tempa\@empty\else:\enskip\fi\relax}%
      \else
        \protected@edef\@svsec{\@seccntformat{##1}\relax}%
      \fi
    \fi
    \@tempskipa ##5\relax
    \ifdim \@tempskipa>\z@
      \begingroup
        \ifnum ##2=\@ne\let\@hangfrom\noindent\fi
        ##6{%
            \@hangfrom{\hskip ##3\relax\@svsec}%
            \interlinepenalty \@M ##8\@@par}%
      \endgroup
      \csname ##1mark\endcsname{##7}%
      \addcontentsline{toc}{##1}{%
        \appendixname\space\csname the##1\endcsname\if!##7!\else:\space\fi%for bookmarks
        ##7}%
    \else
      \def\@svsechd{%
        ##6{\hskip ##3\relax
        \@svsec ##8}%
        \csname ##1mark\endcsname{##7}%
        \addcontentsline{toc}{##1}{%
          \appendixname\space\csname the##1\endcsname\if!##7!\else:\space\fi%for bookmarks
          ##7}}%
    \fi
    \@xsect{##5}}
  \global\let\old@ssect\@ssect
  \def\@ssect##1##2##3##4##5{%
    \@tempskipa ##3\relax
    \ifdim \@tempskipa>\z@
      \begingroup
        ##4{%
          \@hangfrom{\hskip ##1}%
          \@tempskipa##2\relax
          \ifdim\@tempskipa<-17\p@%check for sectionlevel=1
            \appendixname
            \def\@tempa{##5}\ifx\@tempa\@empty\else:\enskip\fi
          \fi
          \interlinepenalty \@M ##5\@@par}%
      \endgroup
    \else
      \def\@svsechd{##4{\hskip ##1\relax ##5}}%
    \fi
    \@xsect{##3}}}
\def\noappendix{\setcounter{secnumdepth}{0}%
  \global\let\section=\orisection
  \global\let\thesection=\theolds
  \global\let\thesubsection=\theoldss
  \ifx\old@sect\@undefined\else\global\let\@sect\old@sect\fi
  \ifx\old@ssect\@undefined\else\global\let\@ssect\old@ssect\fi
  \def\thefigure{\@arabic\c@figure}%
  \def\thelistings{\@arabic\c@listings}%
  \def\thetable{\@arabic\c@table}}
   
   
\def\appendixfigures{%
  \ifx\afterfirstappendixfigures\@undefined
    \let\afterfirstappendixfigures\relax
  \fi
  \setcounter{figure}{0}%
  \def\thefigure{A\@arabic\c@figure}%
  \let\theHfigure\thefigure}
\def\appendixtables{%
  \ifx\afterfirstappendixtables\@undefined
    \let\afterfirstappendixtables\relax
  \fi
  \setcounter{table}{0}%
  \def\thetable{A\@arabic\c@table}%
  \let\theHtable\thetable}
  
\def\appendixfigures{%
    \ifx\afterfirstappendixfigures\@undefined
      \let\afterfirstappendixfigures\relax
      \setcounter{section}{0}%
      \def\thesection{\Alph{section}}%
      \let\theHsection\thesection
    \fi
    \refstepcounter{section}%
    \setcounter{figure}{0}%
    \def\thefigure{\thesection\@arabic\c@figure}%
    \let\theHfigure\thefigure}
  \def\appendixtables{%
    \ifx\afterfirstappendixtables\@undefined
      \let\afterfirstappendixtables\relax
      \setcounter{section}{0}%
      \def\thesection{\Alph{section}}%
      \let\theHsection\thesection
    \fi
    \refstepcounter{section}%
    \setcounter{table}{0}%
    \def\thetable{\thesection\@arabic\c@table}%
    \let\theHtable\thetable}
    
\def\@journalurlInfo{%
  %% URL raus, DOI rein; 2020-05-15
  \if@preface
    %\@journalurl\ifcopyediting\else/\preface@abbrev.pdf\fi
    https://doi.org/10.5194/\preface@abbrev.pdf
  \else
    https://doi.org/10.5194/\if@noref\@msnumber\else\@journalnameshortlower-\@pvol-\@fpage-\@pyear\fi
    %\@journalurl\ifcopyediting\else/\if@noref\@msnumber\else\@pvol/\@fpage/\@pyear\fi/\fi
  \fi}
    
\IfFileExists{url.sty}
  {\RequirePackage{url}\urlstyle{same}}%
  {\CopernicusWarningNoLine{Cannot find url.sty; proceeding without it}%
   \def\url#1{%
     \CopernicusError{To use \string\url, you must have url.sty}{Install url.sty}}}
\newcommand\doitext{https://doi.org/}
\newcommand*\doi{%
  \begingroup
  \lccode`\~=`\#\relax
  \lowercase{\def~{\#}}%
  \lccode`\~=`\_\relax
  \lowercase{\def~{\_}}%
  \lccode`\~=`\<\relax
  \lowercase{\def~{\textless}}%
  \lccode`\~=`\>\relax
  \lowercase{\def~{\textgreater}}%
  \lccode`\~=0\relax
  \catcode`\#=\active
  \catcode`\_=\active
  \catcode`\<=\active
  \catcode`\>=\active
  %catcode change won't work after \bibitem while backref.sty
  %  grabs for the whole entry
  \@doi}
\def\@doi#1{%
  \let\#\relax
  \let\_\relax
  \let\textless\relax
  \let\textgreater\relax
  \edef\x{\toks0={{#1}}}%
  \x
  \edef\#{\@percentchar23}%
  \edef\_{_}%
  \edef\textless{\@percentchar3C}% instead of {\string<} for Apple
  \edef\textgreater{\@percentchar3E}% instead of {\sting>} for Apple
  \edef\x{\toks1={\noexpand\href{https://doi.org/#1}}}%
  \x
  \edef\x{\endgroup\doitext\the\toks1 \the\toks0}%
  \x}
  
\newcommand\urlprefix{}